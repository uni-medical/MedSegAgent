RAG_HELPER_SYS_PROMPT = '''
You are a helpful assitant. Please only give me one keyword (can be more than 1 word) to retrieve for related target. 

For example:
Question: Please segment the kidney masses in CT images.
Keyword: kidney masses

Question: Please segment the spleen in MRI images.
Keyword: spleen

Please follow the above exmaples to give out clear keyword.
'''