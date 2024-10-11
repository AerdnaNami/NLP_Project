from datasets import load_dataset
import pandas as pd
from openai import OpenAI
import time
import re

ds = load_dataset("allenai/qasper")
title = ds['train'][:2]['title']
abstract = ds['train'][:2]['abstract']
data = {'Title':title,'Abstract':abstract}
df = pd.DataFrame(data)

# Initialize OpenAI client
client = OpenAI(api_key='sk-I-OKo5WoYX5FnJMnZhU-JV7RkJFqZNvwBEqHORPug5T3BlbkFJ5xcABHEGB7oPodSzGTM0lEuhIejDvL_nFq-Ejgl4QA')

# List to store results
data = {
    "No": [],
    "Abstract": [],
    "Questions": [],
    "Answers": []
}

# Loop over abstracts and call the OpenAI API for each abstract
for idx, abstract in enumerate(df['Abstract']):
    time.sleep(3)  # To prevent hitting rate limits

    # Call OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Can you generate 7 Question and Answer pairs where the answer is extracted word for word in the given context \n {abstract}"
            }
        ]
    )

    # Extract the generated text
    text = completion.choices[0].message.content

    # Updated regular expression to ensure we capture the last QA pair even if not followed by a newline
    qa_pattern = re.compile(r'\*\*Question:\*\* (.*?)\s+\*\*Answer:\*\* (.*?)(?:\n|$)', re.DOTALL)
    qa_pairs = qa_pattern.findall(text)

    # Append data to the list with structured format
    abstract_no = idx + 1
    if qa_pairs:
        for i, (q, a) in enumerate(qa_pairs):
            data["No"].append(abstract_no)
            # Only add the abstract for the first question-answer pair, leave empty for others
            data["Abstract"].append(abstract if i == 0 else "")
            data["Questions"].append(q)
            data["Answers"].append(a)
    else:
        data["No"].append(abstract_no)
        data["Abstract"].append(abstract)
        data["Questions"].append("")
        data["Answers"].append("")

# Create the final DataFrame
df_final = pd.DataFrame(data)

# Save the DataFrame to Excel with formatting
with pd.ExcelWriter('QA_data.xlsx', engine='xlsxwriter') as writer:
    df_final.to_excel(writer, sheet_name='Sheet1', index=False)

    # Access the workbook and worksheet
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Variables to keep track of row ranges for merging
    current_row = 1  # Start from row 2 in Excel (index 1 because row 1 has headers)
    questions_per_abstract = 7  # You have 7 Q&A per abstract

    # Dynamically calculate the range for merging
    for i in range(len(df_final) // questions_per_abstract):
        # Calculate the start and end row for merging
        start_row = current_row + 1  # Skip header (Excel row 2 for first row of data)
        end_row = start_row + questions_per_abstract - 1

        # Merge cells for "No" and "Abstract"
        worksheet.merge_range(f'A{start_row}:A{end_row}', df_final['No'][start_row-2])  # Merging the "No" column
        worksheet.merge_range(f'B{start_row}:B{end_row}', df_final['Abstract'][start_row-2])  # Merging the "Abstract" column

        # Update the current_row to move to the next abstract block
        current_row += questions_per_abstract

    # Set column widths for better readability
    worksheet.set_column('A:A', 5)   # No column
    worksheet.set_column('B:B', 30)  # Abstract column
    worksheet.set_column('C:C', 30)  # Questions column
    worksheet.set_column('D:D', 30)  # Answers column