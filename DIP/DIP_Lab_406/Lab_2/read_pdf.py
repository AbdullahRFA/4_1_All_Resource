import PyPDF2
import pyttsx3
book = open("scan_01.pdf","rb")
reader = PyPDF2.PdfReader(book)
num_pages = len(reader.pages)
print(num_pages)

first_page = reader.pages[0]
print(first_page.extract_text())

speaker = pyttsx3.init()
speaker.say(first_page.extract_text())
speaker.runAndWait()

# for x in reader.pages:
#     print(x.extract_text())
