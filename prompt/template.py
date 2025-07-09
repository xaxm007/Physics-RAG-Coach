import os
import uvicorn
from typing import List
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# MongoDB connection
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['Prompt_template']
collection = db['prompts']

origins = [
    'http://localhost:8000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptTemplate(BaseModel):
    name: str
    template: str

@app.get('/', response_model=List[PromptTemplate])
async def get_prompts():
    prompts = list(collection.find({}, {"_id": False}))
    return prompts

@app.put('/prompts/{name}', response_model=PromptTemplate)
async def update_prompt(name: str, prompt: PromptTemplate):
    if prompt.name != name:
        raise HTTPException(status_code=400, detail="Prompt name in body must match URL parameter")
    result = collection.update_one(
        {"name": name},
        {"$set": {"template": prompt.template}},
        upsert=True
    )
    if result.matched_count == 0 and result.upserted_id is None:
        raise HTTPException(status_code=404, detail="Prompt not found and not created")
    return {"name": name, "template": prompt.template}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)