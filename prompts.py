
availableactions_k1= "walk, stand, grab, put, putback, drink, open, putin, close, switchon, switchoff "

predprompt = """ 
You will be given {histlen} video frames as history input. The video frames describes humans performing actions. 
Your task is to predict {futrlen} future human actions based on the information provided in the video frames. 
You will be given scene description and scene graph. The scene graph describes the properties of each object in the environment.
The scenegraph of the environment is given below \n {scenegraph} \n\n

The output should be the {futrlen} future action labels. An Action label for one character is '(charid, action, objects)' and if there are two characters the action label is '(charid1, action, objects) (charid2, action, objects)'.
The available charid are: 'female1, male1, female2, male2'.\n
The only valid actions are : '{availableactions}'.\n
The action "put" requires two objects object1 and object2. 
Other actions requires only one object.   

Understand the information from the images and generate {futrlen} future action labels, each action label should be seperated by a comma.
"""

query = """Generate action labels in the suitable format for each character in the provided image. The action label should be in a fixed format.\n
The scene description of the environment is given as {scenedesc}\n\ \n
The only valid actions a character can do are : '{availableactions}'.\n
The available objects a character can interact with in an environment are : '{objects_bed1}'.\n
The available charid are: 'female1, male1'.\n
Use the scene description, action and objects to generate corresponding action for each character in the image.\n
For the action "walk, stand, sit, grab, switchon, switchoff" output should be the format '(charid, action, object1)' for each character.\n
For the action "put" requires two objects, the output format should be '(charid, action, object1, object2)'\n
You cannot add new actions or objects\n
"""

segquery = """You are provided with video frames. Your task is to generate a list of action labels. The action labels describe action performed by each character.\n
You should generate action label for each frame.\n  
The only valid actions a character can do are : '{availableactions}'.\n
The available objects a character can interact with in an environment are : '{objects_bed1}'.\n
The available charid are: 'female1, male1'.\n
An action label should be of the format '(charid, action, object)' for each character.\n
You will be given 4 video frames and the output should be a list of 4 strings. Each string in the list describes the action performed by each character in the image.\n
"""

system_message = """You are expert in recognizing actions from videos."""

