import requests
import json
import sys

data = {
    "content": sys.argv[1]
}


headers = {
    "Content-Type": "application/json",
    "X-LC-Id": "xFIh3rK2UWrwtRPP4WmPrdQ3-MdYXbMMI",
    "X-LC-Key":"fdNQqmMt3W22UQgFyj5MxUld,master",
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36"
 }

## post的时候，将data字典形式的参数用json包转换成json格式。
requests.post(url='https://xFIh3rK2.api.lncldglobal.com/1.1/classes/content', headers=headers, data=json.dumps(data))


""" response = requests.get(url='https://xFIh3rK2.api.lncldglobal.com/1.1/classes/content/5e9e748d2f040b00087abb8b', headers=headers)
response.encoding = 'UTF-8'
response = response.text
print (response) """
