# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# 1. crawl every restaurant in rest.co.il
# 2. extract data into DF
# 3. create new feature "score" based of feature engeeniring
#

# find - finds an html element or tag, and returns the first match.
#
# find_all - returns a list (iterable) of all html elements that match the find criteria.
#
# get_text - returns the human-readable text inside the html elements contained in the BeautifulSoup object.
#
# string - Convenience property of a tag to get the single string within this tag.
#
# prettify - will turn a BeautifulSoup object into a nicely formatted Unicode string, with a separate line for each tag and each string.

from bs4 import BeautifulSoup
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = r"https://www.rest.co.il/restaurants/"
agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
page = requests.get(url, verify=False, headers=agent)

rest_soup = BeautifulSoup(page.content, 'html.parser')

# print(rest_soup.prettify())

text = rest_soup.find("div", attrs={"class":"feature-column-photo", "data-bg":"" ,"data-event-id":"148"})

print(text)