"""
Amadeus AI
Chatbot with; 
Speech to Text, Text to Speech, ASL (American Sign Language) Recognition (Computer Vision) and Local LLM integration

Author: Burak Metin Erdem (Akazawa)
GitHub: https://github.com/AkazawaxQ
Date: 2025

Notes:
- Designed and tested on Linux (Ubuntu 24.04.1 LTS)
- Uses local Ollama (LLaMA 3.1)
    - In order to use this application, you need to have Ollama installed and a local model downloaded.
    - Ollama installation guide: https://ollama.com/docs/installation
    - Download LLaMA 3.1 model: `ollama pull llama3.1`
- ASL recognition trained on a custom dataset created by author.
"""


import tempfile
import tkinter as tk
from tkinter import scrolledtext, font, simpledialog, messagebox
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import threading
import speech_recognition as sr
from TTS.api import TTS  
import re
import atexit
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time

#This is the file where settings and conversations will be stored
DATA_FILE = "app_data.json" 
#That is how we load the YOLO model
modelyolo = YOLO("content-amadeus/runs/detect/train/weights/best.pt")

default_data = {
    "settings": {
        "user_name": "User",
        "character_name": "Bot",
        "character_traits": "",
        "voice": "Woman",
        "theme": "#1e1e1e",
        "side_theme": "#2c2c2c",
        "button_theme": "#444",
        "bubble_color": "#800080",
        "fg_color": "white",
        "speaker": "Claribel Dervla"
    },
    "conversations": {}
}

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("JSON file is broken. Default data will be used.")
    return default_data.copy()

def save_data():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"settings": settings, "conversations": conversations}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving data: {e}")

def save_message(title, message, sender):
    try:
        
        if title not in conversations:
            conversations[title] = [] 

        
        conversations[title].append({"sender": sender, "message": message})

        
        save_data()
    except Exception as e:
        print(f"Error saving message: {e}")

data = load_data()

settings = data["settings"]
conversations = data["conversations"]

template = """
You are a helpful assistant. Continue the conversation naturally without repeating introductions unnecessarily.

Here is the conversation history for context: 
{context}

Based on the conversation so far, answer the following question considering the character traits provided.

Question: {question}

Character Traits: {character_traits}

Answer:
"""
model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

MAX_CONTEXT_LENGTH = 10
context_list = []

character_name = settings["character_name"]
character_traits = settings["character_traits"]

device = "cuda"
tts_thread = None
stop_tts = threading.Event()
#tts_engine_female = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
#tts_engine_male = TTS(model_name="tts_models/en/vctk/fast_pitch")
tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts_engine.to(device)

is_tts_muted = False
def toggle_tts():
    global is_tts_muted
    is_tts_muted = not is_tts_muted
    if is_tts_muted:
        mute_button.config(text="Unmute", bg=settings["bubble_color"])
        stop_tts_function()
    else:
        mute_button.config(text="Mute", bg=settings["button_theme"])

def stop_tts_function():
    global stop_tts
    os.system("pkill aplay")
    stop_tts.set()


def speak_text(text):
    try:
        global stop_tts, speaker, is_tts_muted
        if is_tts_muted:
            return

        stop_tts.clear() 

        filtered_text = re.sub(r"\*.*?\*|\(.*?\)", "", text).strip()

        speaker = settings["speaker"] if settings["voice"] == "Woman" else "Xavier Hayasaka"

        if filtered_text:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                tts_engine.tts_to_file(
                    text=filtered_text,
                    speaker=speaker,
                    language="en",
                    file_path=temp_audio_file.name,
                )

                if stop_tts.is_set() or is_tts_muted:
                    os.remove(temp_audio_file.name)
                    os.system("pkill aplay")
                    return
                os.system(f"aplay {temp_audio_file.name}")
                os.remove(temp_audio_file.name)
    except Exception as e:
        print(f"Error in TTS: {e}")

def speak_text_thread(text):
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()

recognizer = sr.Recognizer()
microphone = sr.Microphone()

recording = False
listen_thread = None

def toggle_recording():
    global recording, listen_thread
    recording = not recording

    if recording:
        speak_button.config(state="normal", bg="grey", text="Stop")
        listen_thread = threading.Thread(target=speech_to_text, daemon=True)
        listen_thread.start()
    else:
        speak_button.config(state="normal", bg=settings["bubble_color"], text="Speak")

def speech_to_text():
    global recording

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        while recording:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                user_input = recognizer.recognize_google(audio, language="en-US")
                user_entry.insert(tk.END, user_input + " ")
            except sr.WaitTimeoutError:
                print("Listening timed out.")
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

def handle_conversation(user_input):
    global context_list

    
    user_context = [entry for entry in context_list if entry.startswith("User:")]

    character_traits = settings["character_traits"]
    
    result = chain.invoke({
        "context": "\n".join(user_context),
        "question": user_input,
        "character_traits": character_traits
    })

    context_list.append(f"User: {user_input}")
    context_list.append(f"{character_name}: {result}")

    if len(context_list) > MAX_CONTEXT_LENGTH * 2:
        context_list = context_list[-MAX_CONTEXT_LENGTH * 2:]

    return result

def send_message():
    global stop_tts
    stop_tts_function()
    os.system("pkill aplay")
    threading.Thread(target=_send_message_logic, daemon=True).start()

def _send_message_logic(*args):
    user_name = settings["user_name"]
    character_name = settings["character_name"]

    user_input = user_entry.get("1.0", tk.END).strip()
    if user_input:
        display_message(f"{user_name}: {user_input}", "user")
        user_entry.delete("1.0", tk.END)
        
        response = handle_conversation(user_input)
        display_message(f"{character_name}: {response}", "bot")

        selected_title = conversation_listbox.get(tk.ACTIVE)
        if not selected_title:
            selected_title = "Default Conversation"
        
        save_message(selected_title, user_input, "user")
        save_message(selected_title, response, "bot")
        
        speak_text_thread(response)

def display_message(message, sender):
    bubble_color = settings["bubble_color"] if sender == "user" else settings["bot_bubble_color"]

    message_frame = tk.Frame(conversation_area, bg=settings["theme"], padx=10, pady=5)
    
    message_label = tk.Label(
        message_frame, 
        text=message, 
        bg=bubble_color, 
        fg=settings["fg_color"], 
        padx=10, 
        pady=5,
        wraplength=conversation_area.winfo_width() // 2,
        justify="left",
        font=custom_font
    )
    message_label.pack(fill="both", expand=True)

    if sender == "user":
        message_frame.pack(anchor="e")
    else:
        message_frame.pack(anchor="w")

    conversation_area.config(state=tk.NORMAL)
    conversation_area.window_create(tk.END, window=message_frame)
    conversation_area.insert(tk.END,"\n")
    conversation_area.yview(tk.END)
    conversation_area.config(state=tk.DISABLED)

def update_conversation_list():
    conversation_listbox.delete(0, tk.END)
    for title in conversations.keys():
        conversation_listbox.insert(tk.END, title)


def create_new_conversation():
    global conversations

    title = simpledialog.askstring("New Conversation", "Enter conversation title:")
    if title:
        title = title.strip()
        if title in conversations:
            messagebox.showerror("Error", "A conversation with this title already exists.")
        elif not title:
            messagebox.showerror("Error", "Conversation title cannot be empty.")
        else:
            conversations[title] = []
            update_conversation_list()
            save_data()

            conversation_listbox.selection_clear(0, tk.END)
            idx = list(conversations.keys()).index(title)
            conversation_listbox.selection_set(idx)
            load_selected_conversation(title)

def delete_selected_conversation():
    selected = conversation_listbox.curselection()
    if selected:
        title = conversation_listbox.get(selected[0])
        if messagebox.askyesno("Delete Conversation", f"Are you sure you want to delete '{title}'?"):
            del conversations[title]
            update_conversation_list()
            save_data()

def load_selected_conversation(title):
    global context_list
    conversation_area.config(state=tk.NORMAL)
    conversation_area.delete("1.0", tk.END)

    if title in conversations:
        context_list = []
        for message in conversations[title]:
            sender = message["sender"]
            content = message["message"]

            if sender == "user":
                display_message(f"{settings['user_name']}: {content}", "user")
                context_list.append(f"User: {content}")
            elif sender == "bot":
                display_message(f"{settings['character_name']}: {content}", "bot")
                context_list.append(f"{settings['character_name']}: {content}")
        if len(context_list) > MAX_CONTEXT_LENGTH * 2:
            context_list = context_list[-MAX_CONTEXT_LENGTH * 2:]

    conversation_area.config(state=tk.DISABLED)

def on_conversation_select(event):
    selected_index = conversation_listbox.curselection()
    if selected_index:  
        selected_title = conversation_listbox.get(selected_index[0])
        load_selected_conversation(selected_title)

def open_settings():
    global settings, tts_engine
    user_name = "User"

    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")

    tk.Label(settings_window, text="Your Username:").pack()
    user_name_entry = tk.Entry(settings_window)
    user_name_entry.insert(tk.END, settings["user_name"])
    user_name_entry.pack()

    tk.Label(settings_window, text="Character Name:").pack()
    character_name_entry = tk.Entry(settings_window)
    character_name_entry.insert(tk.END, settings["character_name"])
    character_name_entry.pack()

    tk.Label(settings_window, text="Character Features:").pack()
    character_traits_entry = tk.Entry(settings_window)
    character_traits_entry.insert(tk.END, settings["character_traits"])
    character_traits_entry.pack()

    tk.Label(settings_window, text="Voice Option:").pack()
    gender_var = tk.StringVar(value=settings["voice"])
    tk.Radiobutton(settings_window, text="Woman", variable=gender_var, value="Woman").pack()
    tk.Radiobutton(settings_window, text="Man", variable=gender_var, value="Man").pack()

    def update_values():
        if theme_var.get() == "#1e1e1e":
            side_theme_var.set("#2c2c2c")
            button_theme_var.set("#444")
            settings["fg_color"] = "white"
            settings["bot_bubble_color"] = "#3b3b3b"
        else:
            side_theme_var.set("#e6e3e6")
            button_theme_var.set("#ccc")
            settings["fg_color"] = "black"
            settings["bot_bubble_color"] = "#d9d9d9"

    tk.Label(settings_window, text="Theme Color:").pack()
    theme_var = tk.StringVar(value=settings["theme"])
    side_theme_var = tk.StringVar(value=settings["side_theme"])
    button_theme_var = tk.StringVar(value=settings["button_theme"])
    tk.Radiobutton(settings_window, text="Light", variable=theme_var, value="#eae8ea", command=update_values).pack()
    tk.Radiobutton(settings_window, text="Dark", variable=theme_var, value="#1e1e1e", command=update_values).pack()

    tk.Label(settings_window, text="Bubble Color:").pack()
    bubble_var = tk.StringVar(value=settings["bubble_color"])
    tk.Radiobutton(settings_window, text="Blue", variable=bubble_var, value="#0084ff").pack()
    tk.Radiobutton(settings_window, text="Green", variable=bubble_var, value="#0bea1e").pack()
    tk.Radiobutton(settings_window, text="Red", variable=bubble_var, value="#ff0000").pack()
    tk.Radiobutton(settings_window, text="Yellow", variable=bubble_var, value="#ffff00").pack()
    tk.Radiobutton(settings_window, text="Purple", variable=bubble_var, value="#800080").pack()
    tk.Radiobutton(settings_window, text="Orange", variable=bubble_var, value="#ffa500").pack()
    tk.Radiobutton(settings_window, text="Pink", variable=bubble_var, value="#ffc0cb").pack()
    tk.Radiobutton(settings_window, text="Grey", variable=bubble_var, value="#808080").pack()
    tk.Radiobutton(settings_window, text="Brown", variable=bubble_var, value="#a52a2a").pack()

    
    def save_settings():
        global speaker
        settings["user_name"] = user_name_entry.get()
        settings["character_name"] = character_name_entry.get()
        settings["character_traits"] = character_traits_entry.get()
        settings["voice"] = gender_var.get()
        settings["theme"] = theme_var.get()
        settings["side_theme"] = side_theme_var.get()
        settings["button_theme"] = button_theme_var.get()
        settings["bubble_color"] = bubble_var.get()
        settings["fg_color"] = "white" if theme_var.get() == "#1e1e1e" else "black"
        settings["speaker"] = "Claribel Dervla" if settings["voice"] == "Woman" else "Xavier Hayasaka"
        save_data()
        root.configure(bg=settings["theme"])
        side_frame.configure(bg=settings["side_theme"])
        button_frame.configure(bg=settings["side_theme"])
        conversation_listbox.configure(bg=settings["side_theme"], fg=settings["fg_color"])
        middle_frame.configure(bg=settings["theme"])
        user_entry_frame.configure(bg=settings["theme"])
        user_entry.configure(bg=settings["side_theme"], fg=settings["fg_color"], insertbackground=settings["fg_color"])
        send_button.configure(bg=settings["bubble_color"], fg=settings["fg_color"])
        speak_button.configure(bg=settings["bubble_color"], fg=settings["fg_color"])
        cam_button.configure(bg=settings["button_theme"], fg=settings["fg_color"])
        settings_button.configure(bg=settings["button_theme"], fg=settings["fg_color"])
        add_button.configure(bg=settings["button_theme"], fg=settings["fg_color"])
        delete_button.configure(bg=settings["button_theme"], fg=settings["fg_color"])
        mute_button.configure(bg=settings["button_theme"], fg=settings["fg_color"])
        conversation_area.configure(bg=settings["theme"])
        right_frame.configure(bg=settings["theme"])
        load_selected_conversation(conversation_listbox.get(tk.ACTIVE))
        settings_window.destroy()

    tk.Button(settings_window, text="Save", command=save_settings).pack()


cap = None  
is_running = False  

def toggle_camera_frame():
    global cap, is_running
    if camera_label.winfo_viewable():
        if cap is not None:
            cap.release()
        cap = None
        is_running = False
        camera_label.pack_forget()
        right_frame.pack_forget()
        cam_button.config(text="Cam On")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Couldn't open camera.")
            cap = None
            return
        is_running = True
        right_frame.pack(side="right", fill="both", padx=10, pady=10)
        camera_label.pack(fill="both", expand=True)
        cam_button.config(text="Cam Off")
        show_frame()

detected_classes = {}

def show_frame():
    global is_running
    if not is_running or cap is None:
        return 

    ret, frame = cap.read()
    if not ret:
        print("Couldn't read frame.")
        return

    
    results = modelyolo(frame, conf=0.25)
    current_time = time.time()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{modelyolo.names[cls]} {conf:.2f}"

            if conf > 0.70:  
                class_name = modelyolo.names[cls]
                if class_name not in detected_classes:
                    detected_classes[class_name] = current_time
                elif current_time - detected_classes[class_name] >= 1:  
                    handle_detected_class(class_name)  
                    detected_classes.pop(class_name, None)  
            else:
                detected_classes.pop(modelyolo.names[cls], None)  

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    
    display_frame = cv2.resize(frame, (640, 480))
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(display_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    
    root.after(10, show_frame)

def handle_detected_class(class_name):
    if class_name == "Enter":
        send_message()
    elif class_name == "Delete":
        user_entry.delete("end-2c", "end")
    elif class_name == "Space":
        user_entry.insert("end", " ")
    else:
        user_entry.insert("end", class_name)

def on_closing():
    global cap
    if cap is not None:
        cap.release()
    root.destroy()

def on_exit():
    stop_tts_function()
    os.system("pkill aplay")



root = tk.Tk()
root.title("Amadeus AI")
root.geometry("800x500")
root.configure(bg=settings["theme"])

custom_font = font.Font(family="Helvetica", size=12)

side_frame = tk.Frame(root, bg=settings["side_theme"], width=300)
side_frame.pack(side="left", fill="y")

button_frame = tk.Frame(side_frame, bg=settings["side_theme"])
button_frame.pack(fill="x")

settings_button = tk.Button(button_frame, text="Settings", command=open_settings, bg=settings["button_theme"], fg=settings["fg_color"])
settings_button.pack(side="left", fill="x", expand=True)

add_button = tk.Button(button_frame, text="+", command=create_new_conversation, bg=settings["button_theme"], fg=settings["fg_color"])
add_button.pack(side="left", fill="x", expand=True)

delete_button = tk.Button(button_frame, text="-", command=delete_selected_conversation, bg=settings["button_theme"], fg=settings["fg_color"])
delete_button.pack(side="left", fill="x", expand=True)

mute_button = tk.Button(button_frame, text="Mute", command=toggle_tts, bg=settings["button_theme"], fg=settings["fg_color"])
mute_button.pack(side="left", fill="x", expand=True)

conversation_listbox = tk.Listbox(side_frame, bg=settings["side_theme"], fg=settings["fg_color"], font=custom_font)
conversation_listbox.pack(fill="both", expand=True)
conversation_listbox.bind("<<ListboxSelect>>", on_conversation_select)

middle_frame = tk.Frame(root, bg=settings["theme"])
middle_frame.pack(side="left", fill="both", expand=True)

conversation_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", width=70, height=20, bg=settings["theme"], fg=settings["fg_color"], bd=0, highlightthickness=0)
conversation_area.pack(in_=middle_frame, padx=10, pady=(10, 0), fill="both", expand=True)

user_entry_frame = tk.Frame(root, bg=settings["theme"])
user_entry_frame.pack(in_=middle_frame, fill="x", padx=10, pady=10)

user_entry = tk.Text(user_entry_frame, height=2, wrap="word", font=custom_font, bg=settings["side_theme"], fg=settings["fg_color"], insertbackground=settings["fg_color"])
user_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))  

send_button = tk.Button(user_entry_frame, text="Send", command=send_message, bg=settings["bubble_color"], fg=settings["fg_color"], font=custom_font)
send_button.grid(row=0, column=1, padx=5)

user_entry.bind("<Return>", lambda event: (send_message(), "break"))

speak_button = tk.Button(user_entry_frame, text="Speak", command=toggle_recording, bg=settings["bubble_color"], fg=settings["fg_color"], font=custom_font)
speak_button.grid(row=0, column=2, padx=5)


cam_button = tk.Button(user_entry_frame, text="Cam On", command=toggle_camera_frame, bg=settings["button_theme"], fg=settings["fg_color"])
cam_button.grid(row=0, column=3, padx=5)

user_entry_frame.grid_columnconfigure(0, weight=1)

right_frame = tk.Frame(root, bg=settings["theme"], width=640)
right_frame.pack(in_=middle_frame, side="right", fill="y")
right_frame.pack_forget()

camera_label = tk.Label(right_frame, bg=settings["theme"], width=640, height=480)
camera_label.pack_forget()

update_conversation_list()
root.mainloop()
atexit.register(on_exit)