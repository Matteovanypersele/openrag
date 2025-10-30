from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()

mount_chainlit(app=app, target="./app_front.py", path="/chainlit")