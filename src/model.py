from paranoid_root.util import get_elements, ele_dir, pre_model_path


if __name__ == '__main__':
    print(pre_model_path)
    elements = get_elements(ele_dir)
    index = 1023
    print(elements[index])
