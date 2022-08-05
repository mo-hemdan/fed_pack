import requests
import json

def post_metrics(monitor_ip, type, ip_addr, round, accuracy, loss, val_accuracy, val_loss):
    
    post_content = json.dumps({
        "type": type, 
        "ip_addr": ip_addr,
        "round": round,
        "Params": {
            "accuracy": accuracy, 
            "loss": loss, 
            "val_accuracy": val_accuracy, 
            "val_loss": val_loss, 
        }
    })
    result = requests.post("http://"+monitor_ip+":5050/post_metrics", data=post_content)
    print("POST: Connection Status: ", result)
    return result

def get_metrics(monitor_ip, type, ip_addr):
    get_params = {
        "type": type, 
        "ip_addr": ip_addr, 
    }
    result = requests.get("http://"+monitor_ip+":5050/get_metrics", params=get_params)
    print("GET: Connection Status: ", result)
    return result.json()

def get_all_metrics(monitor_ip, type, ip_addr):
    result = requests.get("http://"+monitor_ip+":5050/get_all_metrics")
    print("GET: Connection Status: ", result)
    return result.json()

#########################   BASIC EXAMPLE   #############################

# post = True
# get = True
# example_ip = "188.185.11.183"
# # monitor_ip = "127.0.0.1"

# if post:
#     result = post_metrics(
#         monitor_ip = example_ip,
#         type = "server",
#         client_no = None, 
#         round = 12,
#         epoch = 10,
#         accuracy = 0.99, 
#         loss = 467,
#     )
#     print("Content Recieved: ", result.text)

#     result = post_metrics(
#         monitor_ip = example_ip,
#         type = "client",
#         client_no = 5, 
#         round = 5,
#         epoch = 20,
#         accuracy = 0.68, 
#         loss = 689,
#     )
#     print("Content Recieved: ", result.text)

# if get:
#     # GET SERVER Params
#     result = get_metrics(
#         monitor_ip = example_ip,
#         type = "server", 
#         client_no = None,
#     )
#     print("Content Recieved: ", result)

#     # GET Client Params
#     result = get_metrics(
#         monitor_ip = example_ip,
#         type = "client", 
#         client_no = 5,
#     )
#     print("Content Recieved: ", result)