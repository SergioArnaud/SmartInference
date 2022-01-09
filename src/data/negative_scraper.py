'''
Base code obtained from: https://github.com/NIKHILDUGAR/Google-Image-Scraper/blame/master/main.py#L13
Ran locally due to selenium chromedriver requirement
'''
import warnings
warnings.filterwarnings("ignore")
import os
import re
import urllib.request
from selenium import webdriver
from bs4 import BeautifulSoup as soup
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
#from webdriver_manager.chrome import ChromeDriverManager
sear = input('What are you looking for? ')
n_images = int(input('How many images do you want? '))
saveby= int(input("Do you want to save files by item names(enter 0) or ordered numbers(enter 1)? "))
GOOGLE_IMAGE = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'
searchurl = GOOGLE_IMAGE + 'q=' + sear.replace(" ","+")

def init_driver():
    # Returns a webdriver object
    options = webdriver.ChromeOptions()
    options.add_argument("--enable-javascript")
    chrome_options.add_argument("--incognito")
    chrome_driver_binary = "chromedriver"
    driver = webdriver.Chrome(chrome_driver_binary, options=options)
    return driver


print(searchurl)
try:
    driver = init_driver()
except:
    binary: str = r'C:\Program Files\Mozilla Firefox\firefox.exe'
    options = Options()
    # noinspection PyDeprecation
    options.set_headless(headless=True)
    options.binary = binary
    cap = DesiredCapabilities().FIREFOX
    cap["marionette"] = True #optional
    driver = webdriver.Firefox(firefox_options=options, capabilities=cap, executable_path="geckodriver.exe")
driver.get(searchurl)
import time
#if not os.path.exists("/images"):
    #os.mkdir("/images")
save_path =  os.path.join(os.getcwd(), "images")
completeName = os.path.join(save_path,sear)
#if not os.path.exists(completeName):
os.mkdir(completeName)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
try:
    sbutton = driver.find_element_by_class_name("mye4qd")
    sbutton.click()
except:
    print("")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
src=driver.page_source
driver.close()
page_soup = soup(src, "lxml")
j = os.path.join(save_path, sear)
linkcontainers = page_soup.findAll("img", {"class": "rg_i Q4LuWd"})
namecontainers=page_soup.findAll("div", {"class": "bRMDJf islir"})
n_images=min(n_images,len(namecontainers))
print("no. of images available:",len(namecontainers))
print("no. of images to be downloaded:",n_images)
for i in range(n_images):
    link=None
    try:
        link = linkcontainers[i][src]
    except:
        try:
            link = linkcontainers[i]["data-src"]
        except:
            try:
                link = linkcontainers[i]["src"]
            except:
                try:
                    link = linkcontainers[i]["src"][src]["data-src"]
                    break
                except:
                    print(linkcontainers)
    if saveby:
        completeName = os.path.join(j, str(i+1) + ".jpg")
    else:
        name = namecontainers[i].img['alt'].replace("|", ",").replace("\\\\", " ").replace(".", "")
        name = re.sub('[^a-zA-Z0-9 \n\.]', '', name)
        completeName = os.path.join(j, name + ".jpg")
    f = open(completeName, 'wb')
    if link:
        f.write(urllib.request.urlopen(link).read())
    else:
        print(i)
        print("NOT WORKING")
        print("If you are seeing this then email me at nikhil4709@gmail.com Thank you!")
        print(linkcontainers[i])
    f.close()
    print(i+1," Images downloaded till now")
print('Done')
