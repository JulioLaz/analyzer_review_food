from google.generativeai.types import content_types
from google.api_core.exceptions import ResourceExhausted
from collections.abc import Iterable
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def tool_config_from_mode(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )

def classify_sentiment(text: str) -> str:
    """create 1 review of the quality of purchased food. of 10 at 25 words maximum"""

def chat():
    tools = [classify_sentiment]
    tool_config = tool_config_from_mode("none")
    instruction = (
        "Consider that the comments will be graded according to your sentiment."
        "That's why create 1 comment, choose the type of feelings, from positive to negative in different degrees!"
        "25 words maximum"
        "it just returns the clean phrase!"
        "Not absoluty negative nor positive"
        "of 10 at 25 words maximum"
    )
   #  instruction = (
   #      "Consider that the comments will be graded according to your sentiment."
   #      "That's why create 5 comments with different types of feelings, from positive to negative in different degrees!"
   #  )
    model = genai.GenerativeModel("models/gemini-1.5-pro", tools=tools, system_instruction=instruction)
    chat = model.start_chat()
    
    try:
        # Asegúrate de pasar un contenido válido para el argumento 'content'
        content = "Create 1 review of the quality of purchased food choose the type of feelings, ranging from positive to negative. maximum 25 words, not absoluty negative nor positive. of 10 at 25 words maximum. it just returns the clean phrase!"
        respuesta = chat.send_message(content=content, tool_config=tool_config)
        reviews = respuesta.text.strip().split('\n')
        formatted_reviews = '\n'.join([f'"{review.strip()}"' for review in reviews])
        print('Formatted Reviews: \n', formatted_reviews)
        return formatted_reviews

    except ResourceExhausted:
        print("¡Advertencia! Se ha agotado la cuota o el crédito.")
        return "Lo siento, se ha agotado la cuota o el crédito. Por favor, intenta nuevamente más tarde."

    except Exception as e:
        print("Error inesperado:", e)
        return "Lo siento, ha ocurrido un error inesperado."

if __name__ == "__main__":
    chat()
