
import streamlit as st
import openai
import io
import os
from PIL import Image

#from dbgeneric import get_index_for_pdf





# Display image using streamlit
st.image(labcom_logo_preto.jpg, use_column_width='auto')

st.title("RESUMER - Converse com os seus PDFs")


st.write(
    """
RESUMER é uma aplicação do LABCOM/NID para implementar o conceito de CUSTOM AI, ou seja,
assistentes que usam a base particular de documentos do usuário para gerar
seu resultado. Neste caso você fornece o documento em um arquivo PDF e começa a conversar
com a aplicação sobre ele.
"""
)


st.info(
    """
**Para usar é simples:**
1. Você precisa ter uma chave da API da OPEN AI.
2. Insira a chave e dê ENTER no campo abaixo, antes de operar com o seu documento.
3. Depois de inserir a chave, você faz o upload dos documentos que deseja usar.
4. Feito isso você pode começar a fazer suas solicitações e conversar.
5. Como sugestão inicie com algo do tipo: 'Faça um resumo do conteúdo deste documento'
"""
)

#Para recuperar uma chave inserida na configuração do APP use como abaixo"
#openai.api_key = db.secrets.get("OPENAI_API_KEY")
#db.secrets.put("OPENAI_API_KEY", "")
key= st.text_input('Entre com sua chave da OPENAI', type='password')
#db.secrets.put("OPENAI_API_KEY", key)
#openai.api_key = db.secrets.get("OPENAI_API_KEY")
openai.api_key = key
#Para lidar com a chave em outro ambiente fora o Databutton
#os.environ['OPENAI_API_KEY'] = openai_api_key 
#openai.api_key = os.getenv("OPENAI_API_KEY")

# This function feeds PDFs into a vectordb.
@st.cache_data
def create_vectordb(files):
    with st.spinner("Creating your vectordb"):
        vectordb = get_index_for_pdf(files, key)
    st.success("Vectordb criado. Seu Chat está pronto!")
    return vectordb


# Let the user upload pdf files
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

# Or we can fetch one from the Databutton storage
#button = st.button("....or use the Databutton docs")

# If PDF files are uploaded, we create the corresponding vectordb.
# Make sure to also check out the code in the Library.
# Since Streamlit reloads pages all the time, we store the vectordb
# in the sessions state.
if pdf_files:
    st.session_state["vectordb"] = create_vectordb(
        [file.getvalue() for file in pdf_files]
    )


#if button:
    #st.session_state["vectordb"] = create_vectordb([db.storage.binary.get("example-pdf")])


prompt_template = """
    You are a PDF expert who combines the knowledge contained in a PDF with all your other training data. 
    It's your job to help the user consume the content of that PDF by asking questions. 
    Respond objectively and stick to the content of the document.
    {pdf_extract}
"""


# When calling ChatGPT, we  need to send the entire chat history together
# with the instructions. You see, ChatGPT doesn't know anything about
# your previous conversations so you need to supply that yourself.
# Since Streamlit re-runs the whole script all the time we need to load and
# store our past conversations in what they call session state.
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])


# Here we display all messages so far in our convo
for message in prompt:
    # If we have a message history, let's display it
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])


question = st.chat_input("Vamos falar sobre os arquivos que forneceu")
#check_for_openai_key()

# If the user asks a question
if question:
    # First, we load the vectordb. If doesn't exist
    # we ask the user to create it.
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.message("assistant"):
            st.write("Você precisa inserir um PDF")
            st.stop()

    # Then, we look for 5 relevant sections in our vectordb
    search_results = vectordb.similarity_search(question, k=5)
    pdf_extract = "/n ".join([result.page_content for result in search_results])

    # Moreover, we put the pdf extract into our prompt
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Then, we add the user question
    prompt.append({"role": "user", "content": question})

    # And make sure to display the question to the user
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        botmsg = st.empty()  # This enables us to stream the response as it comes

    # Here we call ChatGPT with streaming
    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", messages=prompt, stream=True
    ):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()

            # Let us update the Bot's answer with the new chunk
            botmsg.write(result)

    # When we get an answer back we add that to the message history
    prompt.append({"role": "assistant", "content": result})

    # Finally, we store it in the session state
    st.session_state["prompt"] = prompt
