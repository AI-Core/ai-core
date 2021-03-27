import base64
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("http://alexlenail.me/NN-SVG/AlexNet.html")

canvas = driver.find_element_by_css_selector("#canvas")

# get the canvas as a PNG base64 string
canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)

# decode
canvas_png = base64.b64decode(canvas_base64)

# save to a file
with open(r"canvas.png", 'wb') as f:
    f.write(canvas_png)