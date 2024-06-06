

#  对标签文件进行读取



def check_class_file(filepath):
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return classes


def class_file(filepath):
    return check_class_file(filepath)
