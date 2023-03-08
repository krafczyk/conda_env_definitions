from transformers import pipeline

generator = pipeline('text-generation')
print(generator("In the course, we will teach you how to"))
