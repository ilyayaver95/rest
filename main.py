from bs4 import BeautifulSoup
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = r"https://www.rest.co.il/restaurants/israel"
agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
page = requests.get(url, verify=False, headers=agent)

rest_soup = BeautifulSoup(page.content, 'html.parser')

#print(rest_soup.prettify())

text = rest_soup.find_all("div", attrs={"class":"feature-column"})
print(len(text))