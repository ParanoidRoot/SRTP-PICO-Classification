from src.util import get_elements, ele_dir


if __name__ == '__main__':
    elements = get_elements(ele_dir)
    import random
    index = random.randint(1, 1000)
    print(elements[index])
