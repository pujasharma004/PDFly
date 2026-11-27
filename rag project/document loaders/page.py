from langchain_community.document_loaders import WebBaseLoader

url = "https://www.medicalnewstoday.com/articles/depression-treatment"

data = WebBaseLoader(url)

docs = data.load()

print(len(docs))

