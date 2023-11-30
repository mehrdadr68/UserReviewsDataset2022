from transformers import T5ForConditionalGeneration, T5Tokenizer

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
article = """
Pathetic. No cloud storage of data. No way to restore after backing it up. No good customer support. I'm activating WhatsApp on the same phone after uninstalling. The contact number in my phone is same as the sim and the WhatsApp number. I've the chats backed up on the same Google account which is administering my phone. When I install the app again, it asks to get the permission to Google account but then it asks me to "add account" and not choose one which direct. Facebook gth
"""
# encode the text into tensor of integers using the appropriate tokenizer
inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=512, truncation=True)
# generate the summarization output
outputs = model.generate(
    inputs, 
    max_length=150, 
    min_length=40, 
    length_penalty=2.0, 
    num_beams=4, 
    early_stopping=True)
# just for debugging
#print(outputs)
print(tokenizer.decode(outputs[0]))