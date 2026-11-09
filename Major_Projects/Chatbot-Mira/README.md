# Mira — Gemini Chat

## Run locally

1. Create and activate a virtual environment.
2. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Add a Google AI API key as `GOOGLE_API_KEY` in a `Project/.env` file, or enter it in the app sidebar.
4. Start the app from this directory:

   ```powershell
   streamlit run chatbot.py
   ```

The app keeps conversations in the current Streamlit session. Mira defaults to an upbeat, determined, and encouraging personality, while responding empathetically and realistically to difficult situations. You can adjust the Gemini model, response creativity, and assistant instructions in the sidebar, then export a conversation as text.
