from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
model = SentenceTransformer('all-MiniLM-L6-v2')

# pinecone.init(api_key='23f0f049-c2e6-4407-ac48-26288b2acf33', environment='us-east-1-aws')
# index = pinecone.Index('langchain-chatbot')
# input = "Hello world"
# input_em = model.encode(input).tolist()
# print(type(input_em))
# print(len(input_em))

# question = "Where should I submit my Amma two-wheeler scheme application form?"

# question = "what are the reservation under amma two wheeler scheme for women"

# question = "Give me application form link for Amma Two-Wheeler Scheme"

# question = "Benefits of amma two wheeler scheme"

question = "Eligibility criteria for Uzhavar Padhukappu Thittam"

# question = "Give me loan facilities details for amma two wheeler scheme"

# question = "Loan facilities for the amma two wheeler scheme"

# question = "What is the Eligibility criteria for amma two wheeler scheme?"

pc = Pinecone(api_key="23f0f049-c2e6-4407-ac48-26288b2acf33")
index = pc.Index("rag-chatbot")

def find_match(input):
    input_em = model.encode(input).tolist()
    # result = index.query(input_em, top_k=2, includeMetadata=True)
    response = index.query(
    # namespace="example-namespace",
    vector=input_em,
    top_k=2,
    includeMetadata=True,
    include_values=True
    )
    return response['matches'][0]['metadata']['text']
    # return response


result = find_match(question)
print(result)