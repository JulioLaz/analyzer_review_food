import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.generativeai.types import content_types
from collections.abc import Iterable

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def tool_config_from_mode(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of a given text as positive or negative."""
    
def chat_00(texto):
   tools = [classify_sentiment]
   tool_config = tool_config_from_mode("none")
   instruction = "You are an assistant who can analyze the sentiment of food product comments and estimate a percentage of positive and negative. Just answer how much is positive and how much is negative in percentage, the sum of positive and negative is equal to 100% , in the neutral case it is classified as 50% positive and 50% negative and if it is not classified it returns: 'Sorry, I cannot classify this phrase'."
   model = genai.GenerativeModel("models/gemini-1.5-pro", tools=tools, system_instruction=instruction)
   chat = model.start_chat()
   respuesta = chat.send_message(texto,tool_config=tool_config)
   print('respuesta.tex: ', respuesta.text)
   return respuesta.text

from google.api_core.exceptions import ResourceExhausted

def chat(texto):
    tools = [classify_sentiment]
    tool_config = tool_config_from_mode("none")
    instruction = (
        "You are an assistant who can analyze the sentiment of food product comments "
        "and estimate a percentage of positive and negative. Just answer how much is positive "
        "and how much is negative in percentage, the sum of positive and negative is equal to 100%, "
        "in the neutral case it is classified as 50% positive and 50% negative, and if it is not classified, "
        "it returns: 'Sorry, I cannot classify this phrase'."
    )
    model = genai.GenerativeModel("models/gemini-1.5-pro", tools=tools, system_instruction=instruction)
    chat = model.start_chat()
    
    try:
        # Intento enviar el mensaje
        respuesta = chat.send_message(texto, tool_config=tool_config)
        print('respuesta.text: ', respuesta.text)
        return respuesta.text

    except ResourceExhausted:
        # Manejo del error de cuota agotada
        print("¡Advertencia! Se ha agotado la cuota o el crédito.")
        return "Lo siento, se ha agotado la cuota o el crédito. Por favor, intenta nuevamente más tarde."

    except Exception as e:
        # Manejo de otros posibles errores
        print("Error inesperado:", e)
        return "Lo siento, ha ocurrido un error inesperado."


# texto=input("Escribe una frase para analizar el sentimiento: ")
# chat(texto)