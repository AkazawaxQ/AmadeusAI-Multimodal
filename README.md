# AmadeusAI-Multimodal
<p align="center">
<img width="640" height="360" alt="wallpapernew" src="https://github.com/user-attachments/assets/5e9d0000-6c85-46c4-b93f-e3d528dd1968" />
</p>

AmadeusAI is a **one-of-a-kind multimodal chatbot** combining:  
- **Speech-to-Text (STT)**  
- **Text-to-Speech (TTS)** via Coqui TTS  
- **American Sign Language (ASL) Recognition** (Computer Vision)  
- **Local LLM integration via Ollama** (requires Ollama running locally; cannot be used offline)

> ⚠️ This project was **developed specifically for Linux (Ubuntu 24.04.1 LTS)**.  
> It is **not designed or tested for Windows or Mac**, so compatibility on those platforms is not guaranteed.

---

## Project Overview

AmadeusAI allows natural interaction through **both voice and sign language**.  
The ASL recognition is trained on a **custom dataset of 3,702 annotated images** using **YOLOv11**.  

This project was **entirely developed by a single author**, and at the time of creation (late 2024 – early 2025), it was **one of the first, if not the only, project combining TTS, STT, and ASL recognition** in this way. Ollama did not yet have an official interface publicly available, making this setup truly unique.

## User Interface
![InterfacewithCam](https://github.com/user-attachments/assets/c55fc86c-9e15-4560-9650-a7171b46c11d)



### Features

- **STT & TTS:** Conversational voice input and output using Coqui TTS.  
- **ASL Recognition:** Real-time hand sign detection using YOLOv11.  
- **Local LLM Integration:** Uses Ollama with LLaMA 3.1 for AI responses (requires local Ollama server).  
- **Custom Dataset:** Fully collected and annotated by the author, with 3,702 images.  
- **Single Developer Project:** Designed, implemented, and trained entirely by the author.

---
## Data Structure
<img width="720" height="1280" alt="DataStructure" src="https://github.com/user-attachments/assets/9632f631-c863-4fa5-91ce-dcd09f89c634" />


## Dataset & Model

- **Dataset:** 3,702 annotated images for ASL hand signs.  
- **Training:** YOLOv11 trained on the custom dataset.  
- **Sample Data Included:** Only small sample images and dataset structure for demonstration.  
- **Not Included:** Full dataset, trained model weights, and training logs (available upon request).

---

## Dependencies

All required Python packages are listed in `requirements.txt`.  
Using a virtual environment (venv) is optional but recommended to avoid conflicts with system packages.
To install them:

```bash
pip install -r requirements.txt
