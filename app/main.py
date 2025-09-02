from fastapi import FastAPI

app = FastAPI(title="FastAPI + uv demo")


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI via uv!"}
