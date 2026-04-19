import gradio as gr

def run_studio():
    print("🍐 Little Fig Studio is live at http://0.0.0.0:8888")
    
    def chat_interface(message, history):
        return f"Little Fig Response to: {message}"

    demo = gr.ChatInterface(
        fn=chat_interface,
        title="Little Fig Workspace (CPU)",
        description="Your independent AI fine-tuning environment."
    )
    
    demo.launch(server_name="0.0.0.0", server_port=8888)
