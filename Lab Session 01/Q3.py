def common_elements():
    A=[1,23,4,5,69,100,0,7]
    B=[0,100,1,2,23,4,6,77]
    common = list(set(A) & set(B))
    if  common:
            print(common)
    else:
            print("No common elements")
common_elements()