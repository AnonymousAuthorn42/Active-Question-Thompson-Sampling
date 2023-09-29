import numpy as np
import networkx as nx


#------------------------------------------------------------------------------
# Functions needed to compute questions and hierarchy
#------------------------------------------------------------------------------

def give_last_descendants( node_lvl_dicts, node ):
    '''
    return the leaves corresponding to a node
    function used only in compute_hierarchy
    '''
    if node in node_lvl_dicts[0].keys():
        if len(node_lvl_dicts) == 1 :
            return node_lvl_dicts[0][node]
        else :
            return list(np.concatenate([give_last_descendants( node_lvl_dicts[1:], nodes ) for nodes in node_lvl_dicts[0][node]]) )
    else :
        if len(node_lvl_dicts)>1 :
            return give_last_descendants( node_lvl_dicts[1:], node )
        else :
            return [node]

def compute_hierarchy( node_lvl_dicts, finelabel_to_idx ) :
    ''' 
    node_lvl_dicts : iterable of dictionaries, each of form { parent_node : child_nodes }
                     The list must be sorted in increasing length of dictonaries, and should not contain the root node.
                     
    finelabel_to_idx : dict {label_name : idx}. for example datasets.CIFAR100(*Args).class_to_idx
                      
    parent_node : str
    child_nodes : iterable of str
    '''
    
    for i, dic in enumerate(node_lvl_dicts):
        dic_copy = dic.copy()
        for k, (key, item) in enumerate( dic.items()) :
            if len(item) == 1 and key == item[0] :
                del dic_copy[key]
                dic_copy[key+f'_{k}'] = item
        node_lvl_dicts[i] = dic_copy
    
    intern_nodes = list( np.concatenate([list(dic.keys()) for dic in node_lvl_dicts  ]) )
    leaves = list(finelabel_to_idx.keys())
        
    treeclass_to_ix = {'root' : 0,
                   **{ key : k+1 for k, key in enumerate(intern_nodes+leaves)} }
    
    actualclass_to_treeclass = { item : treeclass_to_ix[key] for key, item in finelabel_to_idx.items()  }
    
    
    tree_to_actual = {0 : [k for k in range(len(finelabel_to_idx))],
                      **{ item : [key] for key, item in actualclass_to_treeclass.items() },
                      **{ treeclass_to_ix[key] : [finelabel_to_idx[name] for name in give_last_descendants( node_lvl_dicts, key )] for key in intern_nodes }
                      }
    
    return treeclass_to_ix, actualclass_to_treeclass, tree_to_actual



def create_graph(node_lvl_dicts, finelabel_to_idx):
    ''' 
    return an nx.Graph corresponding to our hierarchy
    usefull function to save a graph before doing the Graph Sarkar's representation
    '''
    treeclass_to_ix, _, _ = compute_hierarchy(node_lvl_dicts, finelabel_to_idx)
    edges = [(0, treeclass_to_ix[node]) for node in node_lvl_dicts[0]]
    for dic in node_lvl_dicts:
        for key, item in dic.items():
            for elt in item :
                edges.append( (treeclass_to_ix[key],treeclass_to_ix[elt]) )
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])           
    
    nx.write_edgelist(G, 'CIFAR100_tree.edges', data=False)
    
    return G


def random_hierarchy(num_classes, tree_depth, class_names = None ) :
    if class_names is None:
        class_names = [f'class_{k}' for k in range(num_classes)]
    original_class_names = class_names
    original_num_classes = num_classes
    
    hierarchy_node_dicts =[]
    num_nodes = int(num_classes*0.8)
    for k in range(1,tree_depth+1):
        if num_nodes in [0,1,2] :
            break
        rand_num = np.random.randint(1,num_nodes)
        classk_names = [f'node {tree_depth-k}.{i}' for i in range(rand_num)] # class names for lvl k
        partition = np.random.randint(2,num_classes,rand_num)
        count = 0
        while partition.sum() != num_classes :
            count +=1
            try : 
                partition = np.random.randint(2,num_classes//2,rand_num)
            except : 
                partition = np.array([2,2])
            if count>200:
                partition = np.array([2]).repeat(rand_num-1)
                s = partition.sum()
                partition = np.concatenate((partition, np.array([num_classes-s])))
            
        cum_partition = np.concatenate([np.array([0]),np.cumsum(partition)])
        num_nodes = int(rand_num*0.7)   # next upper bound of number of nodes
        
        class_names = list(np.random.permutation(class_names))
        num_classes = rand_num #next number of classes to attribute
        current_dict = { key : class_names[cum_partition[i]:cum_partition[i+1]] for i, key in enumerate(classk_names) }
        class_names = classk_names # next classes to attribute
        hierarchy_node_dicts.append(current_dict)
        
    hierarchy_node_dicts.reverse()
    k=1
    for dictio in hierarchy_node_dicts:
        for key, item in dictio.items():
                k = min(k,len(item))
        
    if k == 0 :
        return random_hierarchy(original_num_classes, tree_depth, original_class_names)
    else :      
        return  hierarchy_node_dicts
        

#------------------------------------------------------------------------------
# CIFAR100 Hierarchy
#------------------------------------------------------------------------------

lvl0_nodes_CIFAR100 = {'man-made_things' : ['man-made_indoor_things',
                                       'large_man-made_outdoor_things_0',
                                       'vehicles'],
                  'natural_things' : ['terrian_mammals',
                                      'large_natural_outdoor_scenes_0',
                                      'insects&reptiles',
                                      'vegetable living',
                                      'aquatic living'
                                      ]}


lvl1_nodes_CIFAR100 = {'man-made_indoor_things': ['food_containers',
                      'household_electrical_devices', 'household_furniture'],
                     'large_man-made_outdoor_things_0': ['large_man-made_outdoor_things'],
                     'large_natural_outdoor_scenes_0' : ['large_natural_outdoor_scenes'],
                     'terrian_mammals': ['large_carnivores','large_omnivores_and_herbivores',
                                          'medium_mammals','people','small_mammals'],
                     'insects&reptiles': ['insects', 'non-insect_invertebrates', 'reptiles'],
                     'vegetable living': ['flowers', 'fruit_and_vegetables', 'trees'],
                     'aquatic living': ['aquatic_mammals', 'fish'],
                     'vehicles': ['vehicles_1', 'vehicles_2']}


lvl2_nodes_CIFAR100 = {'aquatic_mammals'    : ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                         'fish'             : ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                         'flowers'          : ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                         'food_containers'  : ['bottle', 'bowl', 'can', 'cup', 'plate'],
                         'fruit_and_vegetables': ['apple','mushroom','orange',
                                                  'pear','sweet_pepper'],
                         'household_electrical_devices': ['clock','keyboard',
                                                          'lamp','telephone','television'],
                         'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                         'insects'            : ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                         'large_carnivores'   : ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                         'large_man-made_outdoor_things': ['bridge','castle','house',
                                                           'road','skyscraper'],
                         'large_natural_outdoor_scenes': ['cloud','forest','mountain',
                                                          'plain','sea'],
                         'large_omnivores_and_herbivores': ['camel','cattle','chimpanzee',
                                                            'elephant','kangaroo'],
                         'medium_mammals'       : ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                         'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
                         'people'               : ['baby', 'boy', 'girl', 'man', 'woman'],
                         'reptiles'             : ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                         'small_mammals'        : ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                         'trees'                : ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                         'vehicles_1'           : ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                         'vehicles_2'           : ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']} # the original coarse labels

CIFAR100_node_dicts = [lvl0_nodes_CIFAR100,
                    lvl1_nodes_CIFAR100,
                    lvl2_nodes_CIFAR100]

#------------------------------------------------------------------------------

l0 = { 'nature': ['animal', 'flora', 'natural_outdoor_scenes']}

l1 = { 'animal': ['aquatic_animal', 'invertebrate', 'mammal', 'reptile'],
      'man-made_things': ['large_human_construction', 'man-made_object', 'vehicle']}

l2 = { 'mammal': ['aquatic_mammal', 'canine', 'feline',  'large_mammal',  'medium_mammal',  'primate',
                 'small_mammal'],
      'man-made_object': ['container', 'electrical_device', 'household_furniture']}

l3 = { 'primate': ['chimpanzee', 'human'],
       'aquatic_animal': ['amphibian', 'aquatic_mammal', 'fish'],
       'invertebrate': ['insect', 'non_insect_invertebrate', 'shellfish'],
       'flora': ['fruit_and_vegetable', 'plants'],
       'container': ['drink_container', 'food_container'],
       'carnivore': ['bear', 'fox', 'leopard', 'lion', 'shark', 'tiger', 'wolf'],
       'vehicle': ['bus', 'lawn_mower', 'personal_vehicle', 'specific_vehicle']}

l4 = { 'human': ['adult', 'child'],
      'fish': ['large_fish', 'small_fish'],
      'large_human_construction': ['building', 'transport_way'],
      'reptile': ['large_reptile', 'lizard', 'medium_reptile'],
      'insect': ['flying_insect', 'non_flying_insect'],
      'non_insect_invertebrate': ['shellfish', 'spider', 'worm'],
      'amphibian': ['beaver',  'crab',  'crocodile', 'otter', 'seal', 'snake', 'turtle'],
      'fruit_and_vegetable': ['fruit', 'vegetable'],
      'electrical_device': ['clock', 'communication_device', 'keyboard', 'lamp'],
      'personal_vehicle': ['pickup_truck', 'two_wheel_vehicle'],
      'quadruped': ['bear', 'camel','cattle','elephant', 'fox', 'hamster', 'leopard', 'lion',
                    'lizard', 'mouse', 'rabbit', 'shrew', 'squirrel', 'tiger', 'wolf'],
      'household_furniture': ['bed', 'seat', 'table', 'wardrobe'],
      'feeding_furniture': ['drink_container', 'food_container', 'table'],
      'feline': ['leopard', 'lion', 'tiger'],
      'canine': ['fox', 'wolf'],
      'plants': ['flowers', 'forest', 'tree']}

l5 = { 'aquatic_mammal': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
      'small_fish': ['aquarium_fish', 'flatfish', 'trout'],
      'large_fish': ['ray', 'shark'],
      'fruit': ['apple', 'orange', 'pear'],
      'vegetable': ['mushroom', 'sweet_pepper'],
      'child': ['baby', 'boy', 'girl'],
      'adult': ['man', 'woman'],
      'natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
      'tree': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
      'small_mammal': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
      'medium_mammal': ['chimpanzee', 'fox',  'kangaroo', 'leopard', 'porcupine', 'possum',
                        'raccoon', 'skunk', 'wolf'],
      'transport_way': ['bridge', 'road'],
      'building': ['castle', 'house', 'skyscraper'],
      'large_mammal': ['bear', 'camel', 'cattle', 'elephant', 'lion', 'tiger', 'whale'],
      'flying_insect': ['bee', 'beetle', 'butterfly'],
      'non_flying_insect': ['caterpillar', 'cockroach'],
      'shellfish': ['crab', 'lobster', 'snail'],
      'communication_device': ['telephone', 'television'],
      'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
      'drink_container': ['bottle', 'can', 'cup'],
      'food_container': ['bowl', 'plate'],
      'large_reptile': ['crocodile', 'dinosaur'],
      'medium_reptile': ['snake', 'turtle'],
      'two_wheel_vehicle': ['bicycle', 'motorcycle'],
      'trains': ['streetcar', 'train'],
      'specific_vehicle': ['rocket', 'tank', 'tractor'],
      'seat': ['chair', 'couch']}

CIFAR100_node_dicts_2 = [l0,l1,l2,l3,l4,l5]

#------------------------------------------------------------------------------
# CIFAR10 Hierarchy
#------------------------------------------------------------------------------

lvl0_nodes_CIFAR10 = {'living' : ['mammal', 'non-mammal'],
                      'non-living' : ['vehicle', 'craft']}
lvl1_nodes_CIFAR10 = {'mammal' : ['cat','deer','dog','horse'],
                      'non-mammal' : ['bird','frog'],
                      'vehicle' : ['automobile', 'truck'],
                      'craft' : ['airplane', 'ship']}


CIFAR10_node_dicts = [lvl0_nodes_CIFAR10,
                    lvl1_nodes_CIFAR10]

#-----------------------------------------------------------------------------
b={ 'living': ['wild_animal', 'mammal', 'domestic_animal', 'oviparous'], 
   'means_of_transport': ['vehicles', 'craft', 'horse']}

c={ 'mammal': ['large_mammal', 'small_mammal'], 
   'non-living': ['craft', 'vehicles'], 
   'domestic_animal': ['horse', 'pet']}

d={ 'wild_animal': ['deer', 'bird', 'frog']}
e={ 'flying_things': ['airplane', 'bird'], 
   'pet': ['dog', 'cat']}

f={ 'oviparous': ['bird', 'frog'], 
   'small_mammal': ['dog', 'cat'], 
   'large_mammal': ['deer', 'horse'], 
   'vehicles': ['automobile', 'truck'], 'craft': ['ship', 'airplane']}

CIFAR10_node_dicts_2 = [b,c,d,e,f]

#------------------------------------------------------------------------------
CIFAR10_node_dict_3 = [{'living': ['bird', 'cat', 'deer', 'dog', 'frog', 'horse'],
                        'vehicles': ['airplane', 'automobile', 'ship', 'truck']
                        }]
#------------------------------------------------------------------------------
# FashionMNIST Hierarchy
#------------------------------------------------------------------------------

lvl0_nodes_FASHION = {'tops' : ['T-shirt/top', 'Pullover', 'Dress', 'Coat','Shirt'],
                      'bottoms' : ['Trouser'],
                      'Accessories' : ['Bag'],
                      'Footwear' : ['Sandal', 'Sneaker', 'Ankle boot'] }

FASHIONMNIST_node_dicts = [lvl0_nodes_FASHION]

lvl0_node_FASHION_2 = {'top': ['protection_top', 'top_clothes'],
                       'non-top': ['accessories', 'bottom', 'footwear']}

lvl1_node_FASHION_2 = {'accessories': ['Bag'],
                         'protection_top': ['Coat', 'Pullover'],
                         'top_clothes': ['Dress', 'Shirt', 'T-shirt/top'],
                         'bottom': ['Trouser'],
                         'footwear': ['Ankle boot', 'Sandal', 'Sneaker']}

FASHIONMNIST_node_dicts_2 = [lvl0_node_FASHION_2, lvl1_node_FASHION_2]


#------------------------------------------------------------------------------
# TOY and TOY2 Hierarchy
#------------------------------------------------------------------------------


lvl0_node_TOY = {'node 1.1' : ['node 2.1', 'node 2.2'],
                 'node 1.2' : ['node 2.3', 'node 2.4']
                 }
lvl1_node_TOY = {'node 2.1' : ['class_0', 'class_1'],
                 'node 2.2' : [f'class_{k}' for k in range (2,5)],
                 'node 2.3' : ['class_5', 'class_6'],
                 'node 2.4' : [f'class_{k}' for k in range(7,10)]}

TOY_node_dicts = [lvl0_node_TOY,
                  lvl1_node_TOY]

#------------------------------------------------------------------------------
# TinyImageNet 200 Hierarchy
#------------------------------------------------------------------------------
    

lvl0_node_TIN = {'animal_life': ['animal', 'animal_environment'],
                 'man-made_thing': ['human_construction', 'human_activity_related',
                                    'man-made_object', 'tool_or_device', 'vehicle']
                 }

lvl1_node_TIN = {'vehicle': ['boat', 'motored_vehicles', 'unmotored_vehicle'],
                 'animal': ['insectoid', 'mammal', 'non-mammal', 'reptile'],
                 'man-made_object': ['clothes', 'container', 'furnishings',
                                      'outdoor_object',  'poles', 'urban_object'],
                 'tool_or_device': ['devices', 'instrument', 'tool'],
                 'human_activity_related': ['food', 'sport', 'text'],
                 'animal_environment': ['animal_related', 'outdoor_scene']
                 }

lvl2_node_TIN = {'insectoid': ['arachnid', 'insect'],
                 'motored_vehicles': ['cars', 'farm_vehicle', 'train', 'truck'],
                 'reptile': ['big_reptile', 'small_reptile'],
                 'mammal': ['ape', 'aquatic_mammal', 'bears', 'big_mammal',
                            'bovid', 'dog', 'feline',
                            'medium_mammal', 'small_mammal'],
                 'clothes': ['accessories', 'bottom', 'footwear', 'tops'],
                 'unmotored_vehicle': ['unmotored_vehicle'],
                 'container': ['large_container', 'small_container'],
                 'food': ['citrus', 'dish', 'drinks', 'foodstuff', 'non-citrus_fruit', 'vegetable'],
                 'non-mammal': ['amphibians', 'aquatic_non_mammal', 'bird',
                                  'corals',  'lobster', 'mollusk'],
                 'tool': ['construction_object', 'household_object',
                          'medical_object', 'utensil', 'weapon'],
                 'furnishings': ['bathroom_object', 'decoration', 'furniture', 'lamp'],
                 'text': ['board', 'book'],
                 'instrument': ['measuring_instrument', 'musical_instrument', 'optical_instrument'],
                 'devices': ['electrical_device', 'mechanical_device', 'music_device'],
                 'human_construction': ['building', 'large_human_construction',
                                        'shop', 'tower']
                 }

lvl3_node_TIN = {'arachnid': ['black widow', 'scorpion', 'tarantula'],
                 'dog': ['Chihuahua',  'German shepherd', 'Labrador retriever',
                          'Yorkshire terrier', 'cardigan',  'golden retriever',
                          'standard poodle'],
                 'mollusk': ['sea cucumber', 'sea slug', 'slug', 'snail'],
                 'citrus': ['lemon', 'orange'],
                 'feline': ['Egyptian cat', 'Persian cat', 'cougar', 'lion', 'tabby'],
                 'vegetable': ['bell pepper', 'cauliflower', 'mushroom'],
                 'optical_instrument': ['binoculars', 'sunglasses'],
                 'bovid': ['bighorn', 'bison', 'gazelle', 'ox'],
                 'insect': ['bee','centipede', 'cockroach', 'dragonfly', 'fly',
                              'grasshopper', 'ladybug', 'mantis', 'monarch',
                              'sulphur butterfly', 'trilobite', 'walking stick'],
                 'sport': ['basketball', 'dumbbell',  'punching bag', 'rugby ball', 'volleyball'],
                 'shop': ['barbershop', 'butcher shop', 'confectionery'],
                 'non-citrus_fruit': ['acorn', 'banana', 'pomegranate'],
                 'utensil': ['broom', 'teapot', 'wok', 'wooden spoon', 'plate'],
                 'footwear': ['sandal', 'sock'],
                 'dish': ['frying pan', 'pizza', 'potpie'],
                 'foodstuff': ['guacamole',  'ice cream', 'ice lolly',  'mashed potato', 
                               'meat loaf', 'pretzel'],
                 'ape': ['baboon', 'chimpanzee', 'orangutan'],
                 'measuring_instrument': ['hourglass', 'magnetic compass', 'stopwatch'],
                 'lamp': ['candle', 'lampshade', 'torch'],
                 'bird': ['albatross', 'black stork', 'crane', 'goose', 'king penguin'],
                 'musical_instrument': ['drumstick', 'oboe', 'organ'],
                 'small_reptile': ['European fire salamander'],
                 'big_reptile': ['American alligator', 'boa constrictor'],
                 'large_container': ['barrel', 'bucket', 'chest'],
                 'small_container': ['beer bottle', 'pill bottle', 'pop bottle', 'water jug', 'beaker'],
                 'amphibians': ['bullfrog', 'tailed frog'],
                 'corals': ['brain coral', 'coral reef'],
                 'drinks': ['espresso'],
                 'aquatic_mammal': ['dugong'],
                 'cars': ['beach wagon', 'convertible',  'go-kart',
                          'limousine', 'police van', 'sports car'],
                 'bears': ['brown bear', 'lesser panda'],
                 'big_mammal': ['African elephant', 'Arabian camel'],
                 'music_device': ['CD player', 'iPod'],
                 'large_human_construction': ['dam',  'steel arch bridge',  'suspension bridge',
                                              'triumphal arch', 'viaduct', 'water tower'],
                 'truck': ['moving van', 'school bus', 'trolleybus'],
                 'medical_object': ['neck brace', 'syringe'],
                 'small_mammal': ['guinea pig', 'koala'],
                 'unmotored_vehicle': ['jinrikisha'],
                 'accessories': ['academic gown', 'apron', 'backpack',  'bow tie',
                                  'gasmask', 'snorkel', 'sombrero', 'umbrella'],
                 'tops': ['fur coat', 'kimono', 'military uniform', 'poncho', 'vestment'],
                 'tower': ['beacon', 'obelisk'],
                 'urban_object': ['cash machine', 'parking meter'],
                 'building': ['barn', 'thatch'],
                 'furniture': ['desk', 'dining table', 'rocking chair'],
                 'outdoor_scene': ['alp', 'cliff', 'cliff dwelling', 'lakeside', 'seashore'],
                 'household_object': ['plunger', 'teddy'],
                 'mechanical_device': ['abacus', 'chain',  'lawn mower', "potter's wheel",
                                      'reel', 'sewing machine', 'turnstile'],
                 'lobster': ['American lobster', 'spiny lobster'],
                 'medium_mammal': ['hog'],
                 'bottom': ['bikini', 'miniskirt', 'swimming trunks'],
                 'electrical_device': ['computer keyboard', 'pay-phone', 'refrigerator',
                                      'remote control', 'space heater'],
                 'boat': ['gondola', 'lifeboat'],
                 'construction_object': ['nail'],
                 'outdoor_object': ['picket fence'],
                 'weapon': ['cannon', 'projectile'],
                 'poles': ['bannister', 'flagpole', 'maypole', 'pole'],
                 'train': ['bullet train', 'freight car'],
                 'aquatic_non_mammal': ['goldfish', 'jellyfish'],
                 'board': ['brass', 'scoreboard'],
                 'decoration': ['Christmas stocking', 'altar', 'fountain'],
                 'bathroom_object': ['bathtub'],
                 'farm_vehicle': ['tractor'],
                 'animal_related': ['birdhouse', 'spider web'],
                 'book': ['comic book']
                 }
                
TIN_node_dict = [lvl0_node_TIN,
                 lvl1_node_TIN,
                 lvl2_node_TIN,
                 lvl3_node_TIN]
    
#------------------------------------------------------------------------------


lvl0 = {'placental': ['dugong', 'guinea pig', 'ungulate', 'carnivore', 'ape', 'African elephant'],
        'wheeled_vehicle': ['freight car', 'self-propelled_vehicle', 'jinrikisha'],
        'lepidopterous_insect': ['sulphur butterfly', 'monarch'],
        'vessel': ['barrel', 'bathtub', 'beaker', 'bucket', 'water tower', 'teapot', 'bottle'],
        'seabird': ['king penguin', 'albatross'], 'matter': ['solid', 'substance'],
        'cat': ['cougar', 'domestic_cat'], 'timer': ['stopwatch', 'parking meter'],
        'pan': ['wok', 'frying pan'],
        'animal_life': ['animal_environment', 'animal'],
        'object': ['geological_formation', 'whole'],
        'man-made_thing': ['tool_or_device', 'vehicle', 'human_activity_related', 'human_construction', 'man-made_object'],
        'dictyopterous_insect': ['mantis', 'cockroach'],
        'spider': ['black widow', 'tarantula'],
        'solanaceous_vegetable': ['mashed potato', 'bell pepper']} 

lvl1 = {'animal_environment': ['outdoor_scene', 'animal_related'],
        'animal': ['mammal', 'insectoid', 'reptile', 'non-mammal'],
        'tool_or_device': ['tool', 'devices', 'instrument'],
        'self-propelled_vehicle': ['motor_vehicle', 'tractor'],
        'ungulate': ['hog', 'bovid', 'Arabian camel'],
        'whole': ['living_thing', 'artifact', 'natural_object'],
        'human_activity_related': ['sport', 'text', 'food'],
        'carnivore': ['brown bear', 'domestic_animal', 'lesser panda', 'feline'],
        'human_construction': ['large_human_construction', 'tower', 'establishment', 'building'],
        'geological_formation': ['cliff', 'natural_elevation', 'shore'],
        'man-made_object': ['container', 'clothes', 'poles', 'outdoor_object', 'urban_object', 'furnishings'],
        'solid': ['starches', 'produce'], 'bottle': ['pill bottle', 'water jug', 'pop bottle', 'beer bottle'],
        'substance': ['nutriment', 'espresso', 'foodstuff']} 

lvl2 = {'natural_elevation': ['coral reef', 'alp'],
        'large_human_construction': ['viaduct', 'dam', 'triumphal arch', 'steel arch bridge', 'water tower', 'suspension bridge'],
        'clothes': ['accessories', 'tops', 'bottom', 'footwear'], 'starches': ['mashed potato', 'baked_goods'],
        'nutriment': ['dish', 'course'], 'sport': ['dumbbell', 'rugby ball', 'punching bag', 'basketball', 'volleyball'],
        'natural_object': ['acorn', 'edible_fruit'], 'urban_object': ['parking meter', 'cash machine'],
        'living_thing': ['chordate', 'invertebrate'],
        'outdoor_scene': ['cliff', 'lakeside', 'seashore', 'alp', 'cliff dwelling'],
        'produce': ['vegetable', 'edible_fruit'],
        'shore': ['lakeside', 'seashore'],
        'food': ['drinks', 'vegetable', 'dish', 'foodstuff', 'citrus', 'non-citrus_fruit'],
        'outdoor_object': ['picket fence'],
        'poles': ['maypole', 'pole', 'flagpole', 'bannister'],
        'animal_related': ['spider web', 'birdhouse'],
        'insectoid': ['insect', 'arachnid'],
        'tower': ['obelisk', 'beacon'],
        'devices': ['mechanical_device', 'electrical_device', 'music_device'],
        'furnishings': ['lamp', 'decoration', 'furniture', 'bathroom_object'],
        'artifact': ['teddy', 'instrumentality', 'commodity', 'covering', 'structure'],
        'motor_vehicle': ['car', 'go-kart', 'truck'],
        'domestic_animal': ['canine', 'domestic_cat'],
        'non-mammal': ['mollusk', 'amphibian', 'corals', 'aquatic_non_mammal', 'bird', 'lobster'],
        'text': ['book', 'board'],
        'building': ['barn', 'thatch']} 

lvl3 = {'aquatic_non_mammal': ['jellyfish', 'goldfish'],
        'dish': ['pizza', 'frying pan', 'potpie'],
        'mechanical_device': ['reel', 'abacus', "potter's wheel", 'sewing machine', 'chain', 'lawn mower', 'turnstile'],
        'board': ['brass', 'scoreboard'],
        'electrical_device': ['remote control', 'computer keyboard', 'refrigerator', 'space heater', 'pay-phone'], 
        'book': ['comic book'], 
        'structure': ['barn', 'obelisk', 'bridge', 'beacon', 'scoreboard', 'obstruction', 'memorial', 'altar', 'cliff dwelling', 'establishment', 'fountain'], 
        'bathroom_object': ['bathtub'], 
        'foodstuff': ['pretzel', 'ice cream', 'meat loaf', 'ice lolly', 'mashed potato', 'guacamole'], 
        'invertebrate': ['mollusk', 'sea cucumber', 'arthropod', 'coelenterate'], 
        'bottom': ['bikini', 'miniskirt', 'swimming trunks'], 
        'chordate': ['amphibian', 'mammal', 'reptile', 'goldfish', 'bird'], 
        'course': ['dessert', 'plate'], 
        'baked_goods': ['pretzel', 'meat loaf'], 
        'decoration': ['altar', 'Christmas stocking', 'fountain'], 
        'vegetable': ['cauliflower', 'bell pepper', 'mushroom'], 
        'music_device': ['CD player', 'iPod'], 
        'canine': ['standard poodle', 'Chihuahua', 'German shepherd', 'hunting_dog'], 
        'lamp': ['candle', 'lampshade', 'torch'], 
        'accessories': ['apron', 'sombrero', 'bow tie', 'academic gown', 'snorkel', 'umbrella', 'gasmask', 'backpack'], 
        'corals': ['brain coral', 'coral reef'], 
        'instrumentality': ['equipment', 'container', 'furniture', 'conveyance', 'comic book', 'chain', 'implement', 'device'], 
        'drinks': ['espresso'], 'tops': ['fur coat', 'poncho', 'vestment', 'kimono', 'military uniform'], 
        'commodity': ['clothing', 'durables'], 
        'car': ['sports car', 'convertible', 'limousine', 'beach wagon'],
        'covering': ['protective_covering', 'clothing', 'sandal'],
        'edible_fruit': ['banana', 'pomegranate', 'citrus'],
        'non-citrus_fruit': ['acorn', 'banana', 'pomegranate']} 

lvl4 = {'obstruction': ['picket fence', 'dam', 'turnstile', 'bannister'], 
        'clothing': ['apron', 'sombrero', 'garment', 'outerwear', 'footwear', 'military uniform'], 
        'bird': ['albatross', 'goose', 'king penguin', 'crane', 'black stork'], 
        'protective_covering': ['thatch', 'gasmask', 'shelter', 'lampshade'], 
        'arthropod': ['trilobite', 'arachnid', 'centipede', 'lobster', 'insect'], 
        'mollusk': ['sea cucumber', 'sea slug', 'snail', 'slug'], 
        'amphibian': ['frog', 'European fire salamander'], 
        'equipment': ['sports_equipment', 'game_equipment', 'electronic_equipment'], 
        'container': ['large_container', 'small_container'], 
        'bridge': ['steel arch bridge', 'viaduct', 'suspension bridge'], 
        'memorial': ['brass', 'triumphal arch'], 
        'implement': ['utensil', 'pole', 'stick', 'tool', 'broom'], 
        'citrus': ['orange', 'lemon'], 
        'hunting_dog': ['sporting_dog', 'Yorkshire terrier'], 
        'conveyance': ['vehicle', 'public_transport'], 
        'dessert': ['ice cream', 'ice lolly'], 
        'device': ['remote control', 'source_of_illumination', 'musical_instrument', 'spider web', 'computer keyboard', 
                   'nail', 'machine', 'crane', 'snorkel', 'mechanism', 'space heater', 'instrument', 'support'], 
        'establishment': ['confectionery', 'barbershop', 'butcher shop'], 
        'coelenterate': ['brain coral', 'jellyfish'], 
        'durables': ['refrigerator', 'sewing machine'], 
        'mammal': ['aquatic_mammal', 'bears', 'big_mammal', 'dog', 'bovid', 'ape', 'feline', 'medium_mammal', 'small_mammal'], 
        'furniture': ['rocking chair', 'dining table', 'desk'], 
        'reptile': ['big_reptile', 'small_reptile']} 

lvl5 = {'stick': ['flagpole', 'drumstick'], 
        'small_reptile': ['European fire salamander'], 
        'source_of_illumination': ['candle', 'torch'], 
        'big_reptile': ['American alligator', 'boa constrictor'], 
        'big_mammal': ['cougar', 'brown bear', 'lion', 'ape', 'bovid', 'African elephant', 'dugong', 'Arabian camel'],
        'small_container': ['beaker', 'pill bottle', 'water jug', 'pop bottle', 'beer bottle'], 
        'arachnid': ['scorpion', 'black widow', 'tarantula'], 
        'frog': ['bullfrog', 'tailed frog'], 
        'tool': ['utensil', 'household_object', 'construction_object', 'weapon', 'medical_object'],
        'support': ['maypole', 'neck brace'], 
        'insect': ['fly', 'bee', 'cockroach', 'mantis', 'centipede', 'grasshopper', 'trilobite', 'dragonfly', 'ladybug',
                     'walking stick', 'monarch', 'sulphur butterfly'], 
        'shelter': ['umbrella', 'birdhouse'], 
        'sporting_dog': ['Labrador retriever', 'golden retriever'], 
        'medium_mammal': ['domestic_cat', 'dog', 'lesser panda', 'hog'], 
        'aquatic_mammal': ['dugong'], 
        'sports_equipment': ['dumbbell', 'basketball'], 
        'bears': ['brown bear', 'lesser panda'], 
        'electronic_equipment': ['computer keyboard', 'CD player', 'pay-phone', 'iPod'], 
        'large_container': ['chest', 'barrel', 'bucket'], 
        'machine': ['sewing machine', 'abacus', 'cash machine'], 
        'public_transport': ['bullet train', 'bus'], 
        'instrument': ['measuring_instrument', 'optical_instrument', 'musical_instrument'], 
        'footwear': ['sock', 'sandal'], 
        'garment': ['swimsuit', 'overgarment', 'cardigan', 'bow tie', 'miniskirt', 'kimono'], 
        'vehicle': ['motored_vehicles', 'boat', 'unmotored_vehicle'], 
        'feline': ['tabby', 'cougar', 'lion', 'Egyptian cat', 'Persian cat'], 
        'small_mammal': ['koala', 'guinea pig'], 
        'mechanism': ['reel', "potter's wheel"], 
        'lobster': ['spiny lobster', 'American lobster'], 
        'game_equipment': ['rugby ball', 'punching bag', 'basketball', 'volleyball'], 
        'outerwear': ['academic gown', 'vestment']} 

lvl6 = {'utensil': ['wok', 'wooden spoon', 'teapot', 'plate', 'broom'],
        'swimsuit': ['bikini', 'swimming trunks'],
        'bus': ['trolleybus', 'school bus'],
        'household_object': ['plunger', 'teddy'],
        'domestic_cat': ['tabby', 'Egyptian cat', 'Persian cat'],
        'musical_instrument': ['organ', 'drumstick', 'oboe'],
        'optical_instrument': ['binoculars', 'sunglasses'],
        'measuring_instrument': ['stopwatch', 'hourglass', 'magnetic compass'],
        'overgarment': ['fur coat', 'poncho'],
        'boat': ['lifeboat', 'gondola'],
        'dog': ['Labrador retriever', 'cardigan', 'standard poodle', 'Chihuahua', 'Yorkshire terrier', 'German shepherd', 'golden retriever'],
        'construction_object': ['nail'],
        'weapon': ['projectile', 'cannon'],
        'medical_object': ['syringe', 'neck brace'],
        'ape': ['chimpanzee', 'baboon', 'orangutan'],
        'bovid': ['bison', 'ox', 'gazelle', 'bighorn'],
        'motored_vehicles': ['farm_vehicle', 'truck', 'train', 'cars'],
        'unmotored_vehicle': ['jinrikisha']} 

lvl7 = {'train': ['freight car', 'bullet train'],
        'cars': ['sports car', 'convertible', 'go-kart', 'police van', 'limousine', 'beach wagon'],
        'farm_vehicle': ['tractor'],
        'truck': ['trolleybus', 'school bus', 'moving van']} 



TIN_node_dict_2 = [lvl0, lvl1, lvl2, lvl3, lvl4, lvl5, lvl6, lvl7]
    
    

