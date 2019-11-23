import io
import binascii
import urllib.parse
import urllib.request
import urllib.response
 
class DatasetAuthProxy(urllib.request.ProxyHandler):
        
    def __init__(self):
        self.token = 'unknown'
        print( 'token: ' + self.token)
        
    def blue_request(self, req):   #<protocole>_request
        print('auth request MyAuthProxy')
        return req
    
    def blue_response(self, req, response):  #<protocole>_response
        print('auth response MyAuthProxy')
        return response
            
            
class BlueSchemeHandler(urllib.request.BaseHandler):
        
    def __init__(self):
        self.token = 'unknown'
                
    def blue_open(self, req):   #<protocole>_open
        print('Blue BlueSchemeHandler')
        url = req.get_full_url()
        scheme, data = url.split(':', 1)
            
        headers = {}
        newURL = 'https:' + data
        #newURL = urllib.parse.unquote_to_bytes(newURL)
        newReq = urllib.request.Request(newURL)
        fp = urllib.request.urlopen(newReq)
    
        return urllib.response.addinfourl( fp, headers, url)
    
    
myAuthProxy =  DatasetAuthProxy()
blueSchemeHandler = BlueSchemeHandler()
opener = urllib.request.build_opener( myAuthProxy, blueSchemeHandler)
urllib.request.install_opener(opener)


# Client side test - With the \blue\ protocole / scheme \n,
#response = urllib.request.urlopen('blue://www.lefigaro.fr')
#print('Response:\n')
#print(response.read())

f = open('blue://www.lefigaro.fr')
print('Response:\n')
print(response.read())

