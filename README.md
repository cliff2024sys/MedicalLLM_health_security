##You can Clone (or copy) the project
cd ~/path/to/your/projects
git clone <your-repo-url> LLMSecurityResearch
cd LLMSecurityResearch
#########



python3 -m venv venv
source venv/bin/activate     # on macOS/Linux

pip install -r requirements.txt
pip install python-dotenv requests transformers sentence-transformers scikit-learn
pip install torch torchvision 

nano .env
OPENAI_API_KEY="sk-…your OpenAI key…"
HF_API_TOKEN="hf_…your Hugging Face token…"

#run
python updatedmain3.py



