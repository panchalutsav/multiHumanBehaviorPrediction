import copy

BEDROOM1 = {
    "name": "bedroom1", 
    "objects": {
        "closet": {
            "properties": ["CONTAINER"],
            "states": ["OPEN"] 
        },
        "curtains": {
            "properties": [],
            "states": ["OPEN"] 
            
        },
        "bed": {
            "properties": ["SITTABLE"], 
            "states": []
        }, 
        "tablelamp1": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHOFF"]
        },
        "tablelamp2": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHOFF"]
        },
        "chair": {
            "properties": ["SITTABLE"], 
            "states": []
        }, 
        "bookshelf": {
            "properties": ["CONTAINER"], 
            "states": []
        },
        "coffee_table": {
            "properties":["SURFACES"], 
            "states": []
        }, 
        "clothespants": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "class_name": "clothes", 
            "object_placing": {
                "destination": "bed", 
                "relation": "ON"
            }
        },
        "clothesshirt": {
            "properties": ["GRABBABLE"],
            "states": [],  
            "class_name": "clothes",
            "object_placing": {
                "destination": "bed", 
                "relation": "ON"
            }
        },
        "clothespile": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "class_name": "clothes",
            "object_placing": {
                "destination": "bed", 
                "relation": "ON"
            }
        },
    }
}

BEDROOM2 = {
    "name": "bedroom2", 
    "objects": {
        "bed": {
            "properties": ["SITTABLE"], 
            "states": []
        }, 
        "nightlamp": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHOFF"]
        },
        "curtains": {
            "properties": [],
            "states": ["OPEN"] 
        },
        "chair": {
            "properties": ["SITTABLE"], 
            "states": [] 
        }, 
        "bookshelf": {
            "properties": ["CONTAINER"], 
            "states": [] 
        },
        "coffee_table": {
            "properties":["SURFACES"], 
            "states": [] 
        }, 
        "book": {
            "properties": ["GRABBABLE"], 
            "class_name": "book",
            "object_placing": {
                "destination": "bed", 
                "relation": "ON"
            }
        },
        "folder": {
            "properties": ["GRABBABLE"], 
            "class_name": "book",
            "object_placing": {
                "destination": "bed", 
                "relation": "ON"
            }
        },
        "magazine": {
            "properties": ["GRABBABLE"], 
            "class_name": "book",
            "object_placing": {
                "destination": "bed", 
                "relation": "ON"
            }
        },
        "journal": {
            "properties": ["GRABBABLE"], 
            "class_name": "book",
            "object_placing": {
                "destination": "bed", 
                "relation": "ON"
            }
        },
        "chinesebox": {
            "properties": ["GRABBABLE"], 
            "object_placing": {
                "destination": "coffee_table", 
                "relation": "ON"
            }
        },
        "flowerpot": {
            "properties": ["GRABBABLE"], 
            "object_placing": {
                "destination": "coffee_table", 
                "relation": "ON"
            }
        },
    }
}

KITCHEN1 = {
    "name": "kitchen1", 
    "objects": {
        "refridgerator": {
            "properties": ["CAN_OPEN"], 
            "states": ["CLOSED"]
        }, 
        "microwave": {
            "properties": ["CAN_OPEN", "HAS_SWITCH"], 
            "states": ["CLOSED", "SWITCHED_OFF"], 
            "object_placing": {
                "destination": "kitchen_counter", 
                "relation": "ON"
            }
        }, 
        "salmon": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "microwave", 
                "relation": "ON"
            }
        }, 
        "kitchencounter": {
            "properties": [], 
            "states": []
        },
        "kitchencabinet": {
            "properties": ["CAN_OPEN"], 
            "states": ["CLOSED"]
        },
        "sink": {
            "properties": [], 
            "states": []
        }, 
        "faucet": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHED_OFF"], 
            "object_placing": {
                "destination": "sink", 
                "relation": "ON"
            }
        }, 
        "dishwashingliquid": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchencounter", 
                "relation": "ON"
            }
        }, 
        "toaster": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHED_OFF"], 
            "object_placing": {
                "destination": "kitchencounter", 
                "relation": "ON"
            }
        }, 
        "stove": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHED_OFF"], 
        }, 
        "fryingpan": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "stove", 
                "relation": "ON"
            }
        }, 
        "kitchentable": {
            "properties": [], 
            "states": []
        }, 
        "plate": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        },
        "cutleryknife": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        },
        "cutleryfork": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        },
        "cutleryspoon": {
            "properties": ["GRABBABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        },
        "glass": {
            "properties": ["GRABBABLE", "DRINKABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        },
        "milkshake": {
            "properties": ["GRABBABLE", "DRINKABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        },
        "milk": {
            "properties": ["GRABBABLE", "DRINKABLE"], 
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        },
    }
}

LIVINGROOM1 = {
    "name": "livingroom1", 
    "objects": {
        "tv": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHED_OFF"],
            "object_placing": {
                "destination": "tvtable", 
                "relation": "ON"
            }
        }, 
         "remotecontrol": {
            "properties": ["GRABBABLE"], 
            "states": [],
            "object_placing": {
                "destination": "tvtable", 
                "relation": "ON"
            }
        }, 
        "computer": {
            "properties": ["HAS_SWITCH"], 
            "states": ["SWITCHED_OFF"],
        }, 
        "sofa": {
            "properties": ["SITTABLE"], 
            "states": [],
        },
        "chair": {
            "properties": ["SITTABLE"], 
            "states": [],
        },
    }
}

LIVINGROOM2 = {
    "name": "livingroom2", 
    "objects": {
        "chair": {
            "properties": ["SITTABLE"],
            "states": []
        }, 
        "sofa": {
            "properties": ["SITTABLE"],
            "states": []
        }, 
        "bookshelf": {
            "properties": ["CONTAINER"],
            "states": []
        }, 
        "computer": {
            "properties": ["HAS_SWITCH"],
            "states": ["SWITCHED_OFF"]
        }, 
        "coffeetable": {
            "properties": [],
        }, 
        "coffeetable": {
            "properties": [],
        }, 
        "mug": {
            "properties": ["GRABBABLE"],
            "object_placing": {
                "destination": "coffeetable", 
                "relation": "ON"
            }
        }, 
        "book": {
            "properties": ["GRABBABLE"],
            "object_placing": {
                "destination": "coffeetable", 
                "relation": "ON"
            }
        }, 
        "folder": {
            "properties": ["GRABBABLE"],
            "object_placing": {
                "destination": "coffeetable", 
                "relation": "ON"
            }
        }, 

    }
}

LIVINGROOM3 = {
    "name": "livingroom3", 
    "objects": {
        "chair": {
            "properties": ["SITTABLE"],
            "states": []
        }, 
        "sofa": {
            "properties": ["SITTABLE"],
            "states": []
        }, 
        "computer": {
            "properties": ["HAS_SWITCH"],
            "states": ["SWITCHED_OFF"]
        }, 
        "tv": {
            "properties": ["HAS_SWITCH"],
            "states": ["SWITCHED_OFF"]
        }, 
        "coffeetable": {
            "properties": [],
        }, 
        "milkshake": {
            "properties": ["GRABBABLE"],
            "object_placing": {
                "destination": "coffeetable", 
                "relation": "ON"
            }
        }, 
    }
}
KITCHEN3 = {
    "name": "kitchen3", 
    "objects": {
        "mug": {
            "properties": ["GRABBABLE"],
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        }, 
        "plate": {
            "properties": ["GRABBABLE"],
            "states": [], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        }, 
        "kitchentable": {
            "properties": [],
            "states": []
        }, 
        "dishwasher": {
            "properties": ["HAS_SWITCH"],
            "states": ["SWITCHED_OFF"]
        }, 
        "stove": {
            "properties": ["HAS_SWITCH"],
            "states": ["SWITCHED_OFF"], 
            "object_placing": {
                "destination": "kitchentable", 
                "relation": "ON"
            }
        }, 
        "cookingpot": {
            "properties": ["GRABBABLE"],
            "states": [], 
            "object_placing": {
                "destination": "stove", 
                "relation": "ON"
            }
        }, 
        "coffeemaker": {
             "properties": ["HAS_SWITCH"],
            "states": ["SWITCHED_OFF"], 
            "object_placing": {
                "destination": "kitchencounter", 
                "relation": "ON"
            }
        }, 
        "sink": {
             "properties": [],
            "states": [], 
            "object_placing": {
                "destination": "kitchencounter", 
                "relation": "ON"
            }
        }, 
        "faucet": {
             "properties": ["HAS_SWITCH"],
            "states": ["SWITCHED_OFF"], 
            "object_placing": {
                "destination": "kitchencounter", 
                "relation": "ON"
            }
        }, 
        "kitchencounter": {
            "properties": [],
            "states": [], 
        }, 
        "kitchencabinet": {
            "properties": ["CONTAINER"],
            "states": ["closed"], 
        }, 
    }
}



def get_scene_graph(scene_name: str):
    return copy.deepcopy(eval(scene_name.upper())) 


def gsg2(scene_name: str):
    return eval(scene_name.upper())

if __name__ == "__main__":
    bedroom1 = get_scene_graph("bedroom1")
    print(bedroom1)


