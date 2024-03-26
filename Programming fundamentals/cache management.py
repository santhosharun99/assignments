# SANTHOSH ARUNAGIRI ID : 201586816

def find_min(d):
    min_in_vals = min(d.values())
    min_key = float("inf")
    for i,j in d.items():
        if j == min_in_vals:
            min_key = min(i,min_key)

    return min_key


def handle_delete(algo_type):
    if algo_type == "lfu":
        least_used = find_min(cache_dict)
        #insert element with count in is_deleted dict and then delete
        is_deleted[least_used] = cache_dict[least_used]
        del cache_dict[least_used]
        
        for i in range(len(cache)):
            if cache[i] == least_used:
                cache.pop(i)
                break
    elif algo_type == "fifo":
        first = cache[0]
        #insert element with count in is_deleted dict and then delete
        is_deleted[first] = cache_dict[first]
        del cache_dict[first]
        cache.pop(0)


def fifo(page):
    if page in cache_dict:
        cache_dict[page] += 1
        print("hit")
        return
    
    if len(cache) >= size:
        handle_delete(fifo.__name__)

    cache.append(page)
    prev_count = is_deleted.get(page,0)
    cache_dict[page] = prev_count + 1
    print('miss')

def lfu(page):
    if page in cache_dict:
        cache_dict[page] += 1
        print("hit")
        return
    
    if len(cache) >= size:
        handle_delete(lfu.__name__)

    cache.append(page)
    prev_count = is_deleted.get(page,0)
    cache_dict[page] = prev_count + 1
    print("miss")

def choose_cache_algo(key):
    cache_algos = {
        '1': fifo,
        '2': lfu
    }

    return cache_algos.get(key)

def is_input_valid(user_input:str):
    return user_input.isdigit() or user_input in exit_keys

def get_user_input():
    user_input = input(">>>")

    while not is_input_valid(user_input):
        print("OOPS! Invalid input")
        print("Enter a valid integer option or press Q to exit program")
        user_input = input(">>>")

    return user_input

def reset_all():
    global cache,requests,cache_dict,is_deleted
    cache = []
    requests = []
    cache_dict = {}
    is_deleted = {}


size = 8
cache = []
requests = []
cache_dict = {}
is_deleted = {}
exit_keys = ['q',"Q"]
game_on = True

print("Enter your page number.............")
while game_on:
    req = get_user_input()

    #for getting request 2
    
    while req != '0':
        if req in exit_keys:
            game_on = False
            break
    
        requests.append(req)
        req = get_user_input()

    if not game_on:
        break
    
    requests = [int(i) for i in requests]
    print("Press 1 for fifo")
    print("press 2 for lfu")
    print("Press Q to quit the program")
    algo_option = get_user_input()
    if algo_option in exit_keys:
        game_on = False
        break

    algo = choose_cache_algo(algo_option)

    if not algo:
        game_on = False
        break
    
    for req in requests:
        algo(req)
    cache = [int(i) for i in cache]

    print(cache)
    print("To start a new program....")
    print("Enter your page number")
    print('Press Q to quit the program')
   
    reset_all()