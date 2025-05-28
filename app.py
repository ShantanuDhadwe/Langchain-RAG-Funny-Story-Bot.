# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import time 

# Langchain and LLM related imports
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough, RunnableMap 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, SystemMessage, AIMessage 

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma


st.set_page_config(page_title="Funny Storyteller Bot", layout="wide", initial_sidebar_state="expanded")

# custom modules
try:
    from audio_handler import speech_to_text_from_mic, text_to_speech_elevenlabs 
    AUDIO_ENABLED = True
    print("Audio handler loaded successfully.")
except ImportError as e:
    print(f"Warning: audio_handler.py not loaded ({e}). Audio features will be disabled.")
    AUDIO_ENABLED = False
    # Dummy functions if audio_handler is missing
    def speech_to_text_from_mic(timeout=5, phrase_time_limit=None): 
        st.sidebar.warning("Audio input (STT) disabled: audio_handler.py not found or has issues.")
        return None 
    def text_to_speech_elevenlabs(text): 
        st.sidebar.warning(f"Audio output (TTS) disabled: audio_handler.py not found or has issues. Bot would say: {text[:50]}...")
        return None

try:
    from image_generation import generate_img 
    IMAGE_GEN_ENABLED = True
    print("Image generation util loaded successfully.")
except ImportError as e:
    print(f"Warning: image_generation.py not loaded ({e}). Image generation will be disabled.")
    IMAGE_GEN_ENABLED = False
    def generate_img(prompt): 
        st.sidebar.warning(f"Image generation disabled: image_generation.py not found. Prompt was: {prompt}")
        return None

# --- Load Environment Variables ---
load_dotenv()
hf_token_loaded_init = os.getenv('HUGGINGFACEHUB_API_TOKEN') is not None
if AUDIO_ENABLED:
    el_token_loaded_init = os.getenv('ELEVENLABS_API_KEY') is not None
else:
    el_token_loaded_init = False


# --- Global Setup (cached by Streamlit for efficiency) ---
@st.cache_resource
def get_embeddings_model():
    print("Attempting to load embeddings model...")
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print("Embeddings model loaded.")
    return model

@st.cache_resource
def load_vector_db(_embeddings):
    print("Attempting to load Chroma DB...")
    curr_dir = os.getcwd()
    persistent_dir = os.path.join(curr_dir, "db", "chroma_db")
    if not os.path.exists(persistent_dir):
        st.error(f"CRITICAL ERROR: Chroma DB directory not found: {persistent_dir}. Please run your data ingestion script first.") 
        return None
    db = Chroma(persist_directory=persistent_dir, embedding_function=_embeddings)
    print("Chroma DB loaded.")
    return db

@st.cache_resource
def get_llm_model():
    print("Attempting to initialize LLM endpoint...")
    if not hf_token_loaded_init: 
        st.error("CRITICAL ERROR: Hugging Face API Token not found in environment. LLM cannot be initialized.")
        return None
    try:
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3", # Or your preferred model
            task="text-generation"
        )
        model = ChatHuggingFace(llm=llm_endpoint)
        print("LLM and ChatModel initialized.")
        return model
    except Exception as e:
        st.error(f"CRITICAL ERROR: Failed to initialize LLM: {e}")
        return None

embeddings = get_embeddings_model()
db = load_vector_db(embeddings)
model = get_llm_model() 

if db is None or model is None:
    st.error("A critical component (Vector DB or LLM) failed to load. The application cannot continue. Please check the console for errors.")
    st.stop() # Stop Streamlit execution if critical components are missing

@st.cache_resource 
def get_retriever(_db_instance):
    print("Attempting to create retriever...")
    if _db_instance is None: 
        # This case should be caught by the check above, but good to have
        st.error("Retriever creation failed: DB instance is None.")
        return None 
    retriever_instance = _db_instance.as_retriever(search_kwargs={"k": 3}) 
    print("Retriever created.")
    return retriever_instance

retriever = get_retriever(db)
if retriever is None:
    st.error("Failed to create retriever. Application cannot continue.")
    st.stop()

# --- Helper to format documents ---
def format_docs(docs):
    if not docs: return "No relevant context was found in our storybooks for your query."
    return "\\n\\n---\\n\\n".join([d.page_content for d in docs])

# --- 1. Relevance Check System Prompt ---
RELEVANCE_SYSTEM_PROMPT_TEXT = """You are a highly precise classification assistant.
Your ONLY task is to determine if a User's Question can be meaningfully answered using ONLY the Provided Book Context.
The context is from classic books: "Alice in Wonderland," "Gulliver's Travels," or "The Arabian Nights."

Strict Output Format:
Your entire response MUST be a single word: YES or NO.
Do NOT add any other text, explanation, or punctuation. Do not output your reasoning.

--- START OF EXAMPLES ---
Example 1:
User's Question: "Tell me about the Mad Hatter."
Provided Book Context:
---
The Hatter was having tea with the March Hare and a Dormouse. He asked Alice a riddle.
---
Can the "User's Question" be answered based ONLY on the "Provided Book Context"?
YES

Example 2:
User's Question: "What is the capital of France?"
Provided Book Context:
---
Alice fell down the rabbit hole.
---
Can the "User's Question" be answered based ONLY on "Provided Book Context"?
NO
--- END OF EXAMPLES ---

--- CURRENT TASK ---"""
# The relevance chain will take "user_query" and "retrieved_context"

# --- 2. Storytelling Prompt Function (incorporates history) ---
def story_prompt_messages_builder(user_query, retrieved_context, chat_history_messages):
    input_prompt_for_story = (
        f"User's Original Query:\n\"{user_query}\"\n\n"
        f"Story Context from our books:\n---\n{retrieved_context}\n---"
    )
    system_message_content = """You are a funny storyteller. You know stories from "Alice in Wonderland," "Gulliver's Travels," and "The Arabian Nights."
Your goal is to make the user laugh.
IMPORTANT: Only use the "Story Context" I give you. Do not use any other knowledge.
If the context is "No relevant context was found...", explain this humorously.

Your Task:
1.  Carefully read the "User's Original Query" and the "Story Context from our books."
2.  Based ONLY on the "Story Context," craft a reply to the user's query. Retell or explain the relevant part of the story in a very FUNNY and ENGAGING tone.
3.  After your story, provide an "Image Idea" based on the story.

Strictly use this output format:
**Your Witty Tale:**
[Your story here.]
**Whimsical Image Idea:**
[Your image description, or "No image appropriate if no story was told."]"""

    messages = [SystemMessage(content=system_message_content)]
    messages.extend(chat_history_messages)
    messages.append(HumanMessage(content=input_prompt_for_story))
    return messages

# --- 3. "I Don't Know" Prompt Function (incorporates history) ---
def idk_prompt_messages_builder(user_query, chat_history_messages):
    system_message_content = """System: You are a funny storyteller. You know stories from "Alice in Wonderland," "Gulliver's Travels," and "The Arabian Nights."
Your goal is to make the user laugh. You always try to be amusing, even when you can't directly answer a question about other topics."""
    
    human_message_content = f"I asked about: \"{user_query}\"\n\nYour Situation: Alas, my dear friend! It seems my ancient scrolls and magical maps don't contain any tales or wisdom about \"{user_query}\". My expertise is strictly in the realms of whimsical wonderlands, peculiar voyages, and enchanted eastern nights.\n\nYour Task: Craft a SHORT and FUNNY \"I don't know...\" type of message. Acknowledge you couldn't find information. Maintain your funny storyteller persona. DO NOT try to answer. Suggest asking about Alice, Gulliver, Sinbad, etc.\n\nOutput Format: Please provide only your funny \"I don't know\" response."
    
    messages = [SystemMessage(content=system_message_content)]
    messages.extend(chat_history_messages)
    messages.append(HumanMessage(content=human_message_content))
    return messages

# --- Memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print("Initialized ConversationBufferMemory in session state.")

# --- Parsing Function for Story/Image Idea ---
def parse_story_response(llm_response_text):
    cleaned_text = llm_response_text.strip()
    story_marker = "**Your Witty Tale:**"
    image_marker = "**Whimsical Image Idea:**"
    story, image = "Could not parse story.", "Could not parse image idea."
    
    img_marker_idx = cleaned_text.find(image_marker)
    story_marker_idx = cleaned_text.find(story_marker)

    if img_marker_idx != -1: 
        image = cleaned_text[img_marker_idx + len(image_marker):].strip()
        story_section_candidate = cleaned_text[:img_marker_idx].strip()
        if story_marker_idx != -1 and story_marker_idx < img_marker_idx:
            story = story_section_candidate[story_marker_idx + len(story_marker):].strip()
        else: story = story_section_candidate 
    elif story_marker_idx != -1: 
        story = cleaned_text[story_marker_idx + len(story_marker):].strip()
        image = "Image idea not found in response."
    else: story, image = cleaned_text, "Response format unexpected."
    return story, image

# --- Streamlit UI ---
st.title("📚🎙️ Funny Storyteller Bot 🖼️")

# Sidebar
with st.sidebar:
    st.header("Controls & Status")
    st.write(f"HF Token: {'✅ Loaded' if hf_token_loaded_init else '❌ Not Found!'}")
    if AUDIO_ENABLED:
        st.write(f"ElevenLabs Token: {'✅ Loaded' if el_token_loaded_init else '❌ Not Found!'}")
        st.write(f"Audio Input/Output: {'✅ Enabled'}")
    else:
        st.write(f"Audio Input/Output: {'❌ Disabled'}")
    st.write(f"Image Generation: {'✅ Enabled' if IMAGE_GEN_ENABLED else '❌ Disabled'}")
    if st.button("Clear Chat History & Memory", key="clear_chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared! How can I amuse you today?"}]
        st.session_state.memory.clear()
        if "audio_to_play" in st.session_state: 
            del st.session_state.audio_to_play
        st.rerun()

# Initialize/Display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me for a funny story from classic tales, or about characters like Alice, Gulliver, or Sinbad!"}]

if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None

# Display existing chat messages and audio play buttons
for i, msg_data in enumerate(st.session_state.messages):
    with st.chat_message(msg_data["role"]):
        st.markdown(msg_data["content"])
        if msg_data["role"] == "assistant":
            if "image_display" in msg_data and msg_data["image_display"] is not None:
                st.image(msg_data["image_display"], caption="Whimsical Scene", use_column_width=True)
            
            if AUDIO_ENABLED and "audio_bytes" in msg_data and msg_data["audio_bytes"] is not None:
                button_key = f"play_audio_{i}_{msg_data['content'][:10]}" # More unique key
                if st.button("▶️ Play Story Audio", key=button_key):
                    st.session_state.audio_to_play = msg_data["audio_bytes"]
                    # No st.rerun() here, let audio_to_play trigger display below

# If audio_to_play is set (by a button click), display the audio player
if st.session_state.audio_to_play:
    st.audio(st.session_state.audio_to_play, format="audio/mp3")
    st.session_state.audio_to_play = None # Clear it after displaying so it doesn't persist without button click

# --- Input Handling ---
user_query_from_input = None

# Using columns for better layout of text input and speak button
input_col, button_col = st.columns([4,1])

with input_col:
    # Using a form for text input to prevent immediate rerun on every keystroke
    with st.form(key="text_input_form", clear_on_submit=True):
        text_in = st.text_input("Your query:", key="text_input_box_val", placeholder="Type or click 'Speak'...", label_visibility="collapsed")
        submitted_text = st.form_submit_button("💬 Send")
        if submitted_text and text_in:
            user_query_from_input = text_in
            
with button_col:
    if AUDIO_ENABLED:
        if st.button("🎤 Speak", key="speak_button", use_container_width=True):
            with st.spinner("Listening carefully..."):
                recognized_text = speech_to_text_from_mic(timeout=5, phrase_time_limit=7) 
            if recognized_text:
                st.success(f"Heard: {recognized_text}")
                user_query_from_input = recognized_text # This will trigger processing below
            elif user_query_from_input is None: 
                st.warning("No speech detected or understood.")
    else:
        st.button("🎤 Speak", key="speak_button_disabled", use_container_width=True, disabled=True, help="Audio input is disabled. Check console.")


# Process query if available (from text submit or audio button)
if user_query_from_input:
    st.session_state.messages.append({"role": "user", "content": user_query_from_input})
    # User message will be displayed on the next rerun, triggered at the end of this block.
    
    with st.spinner("Consulting ancient scrolls and witty spirits... This might take a moment!"):
        # --- Main Query Processing Logic (Simplified handle_user_query) ---
        
        # 1. Initial Retrieval for Relevance Check
        retrieved_docs = retriever.invoke(user_query_from_input)
        context_for_relevance_check_str = format_docs(retrieved_docs)
        
        relevance_human_message_content = (
            f"User's Question: \"{user_query_from_input}\"\n\n"
            f"Provided Book Context:\n---\n{context_for_relevance_check_str}\n---\n\n"
            f"Based ONLY on the \"Provided Book Context,\" can the \"User's Question\" be meaningfully answered?"
        )
        messages_for_relevance = [
            SystemMessage(content=RELEVANCE_SYSTEM_PROMPT_TEXT),
            HumanMessage(content=relevance_human_message_content)
        ]
        
        # 2. LLM Relevance Check
        raw_relevance_output = "NO (Default on error)" # Default
        try:
            ai_message_relevance = model.invoke(messages_for_relevance)
            raw_relevance_output = ai_message_relevance.content 
        except Exception as e:
            st.error(f"LLM Relevance Check Error: {e}")
            # Potentially log full traceback to console:
            # import traceback
            # traceback.print_exc()

        print(f"LLM Raw Relevance Output: '{raw_relevance_output}'")
        
        is_relevant = False
        processed_relevance_text = raw_relevance_output.strip().upper().replace('"', '').replace('.', '')
        if processed_relevance_text == "YES": is_relevant = True
        elif "YES" in processed_relevance_text and "NO" not in processed_relevance_text: is_relevant = True; st.sidebar.info(f"Relevance: YES (heuristic)")
        else: st.sidebar.info(f"Relevance: NO (processed: '{processed_relevance_text}')")
        st.sidebar.caption(f"For query: '{user_query_from_input[:30]}...' -> Relevant: {is_relevant}")

        # 3. Generate final response
        bot_response_text = ""
        image_idea_text = ""
        generated_image_pil = None
        audio_bytes_for_this_response = None

        chat_history_for_prompt = st.session_state.memory.chat_memory.messages

        if is_relevant:
            with st.spinner("Weaving a funny tale..."):
                # For story generation, we need the context that was ALREADY retrieved for the relevance check.
                # No need for rag_chain to re-retrieve if context_for_relevance_check_str is good.
                story_messages = story_prompt_messages_builder(user_query_from_input, context_for_relevance_check_str, chat_history_for_prompt)
                story_response_full = model.invoke(story_messages).content
                bot_response_text, image_idea_text = parse_story_response(story_response_full)
            
            if IMAGE_GEN_ENABLED and image_idea_text and image_idea_text not in ["Could not parse image idea.", "Image idea not found in response.", "Response format unexpected.", "No image appropriate if no story was told."]:
                with st.spinner("Painting a thousand words (or so)..."):
                    try:
                        generated_image_pil = generate_img(image_idea_text)
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")
        else: # Not relevant
            with st.spinner("Consulting the 'Book of Polite Evasions'..."):
                idk_messages_list = idk_prompt_messages_builder(user_query_from_input, chat_history_for_prompt)
                bot_response_text = model.invoke(idk_messages_list).content

        # Generate TTS for the bot's response text
        if AUDIO_ENABLED and bot_response_text:
            with st.spinner("Warming up the vocal cords..."):
                audio_data_io = text_to_speech_elevenlabs(bot_response_text) 
                if audio_data_io:
                    audio_bytes_for_this_response = audio_data_io.getvalue()

        # Prepare assistant message for display and memory
        assistant_message_to_store = {"role": "assistant", "content": bot_response_text}
        if generated_image_pil:
            assistant_message_to_store["image_display"] = generated_image_pil
        if audio_bytes_for_this_response:
            assistant_message_to_store["audio_bytes"] = audio_bytes_for_this_response 
        
        st.session_state.messages.append(assistant_message_to_store)
        st.session_state.memory.save_context({"input": user_query_from_input}, {"output": bot_response_text})
        
        # Clear the pending audio playback state from any *previous* turn's button click
        # This is important because we are about to rerun and display new messages.
        # The new message will have its own play button.
        if "audio_to_play" in st.session_state:
             st.session_state.audio_to_play = None

        st.rerun() # Rerun to update the display with the new user & assistant messages
