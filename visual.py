""" 使用HTML展示图片 
create_date： 2023.10.12
update_date： 2023.10.13
"""
import os

# 根目录
root = '/home/grey/PycharmProjects/pythonProject/Homomorphic filter/result/experiment_1/uneven'
num = 5

def get_subfolder_names(parent_folder):
    """
    获取指定文件夹下的子文件夹名列表
    """
    subfolders = os.listdir(parent_folder)
    return [f for f in subfolders if os.path.isdir(os.path.join(parent_folder, f))]


def get_image_names(root, subfolder_name, num):
    """
    获取子文件夹内的前 num 张图像的文件名
    """
    image_names = []
    subfolder_path = os.path.join(root, subfolder_name)

    for filename in os.listdir(subfolder_path):
        if os.path.isfile(os.path.join(subfolder_path, filename)):
            image_names.append(filename)

    return image_names[:num]


def read_parameters(file_path):
    """
    从文件中读取 alpha, beta, gamma, 和 cutoff 参数并返回参数字符串
    """
    parameters = {}
    with open(file_path, "r") as file:
        lines = file.readlines()

        head, tail = os.path.split(file_path)
        if tail:
            head2, tail2 = os.path.split(head)

        for line in lines:
            parts = line.split("=")
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                parameters[key] = value

    testnum = parameters.get("Process image number", "")
    alpha = parameters.get("alpha", "")
    beta = parameters.get("beta", "")
    gamma = parameters.get("gamma", "")
    cutoff = parameters.get("cutoff", "")

    # 将参数转化为字符串
    parameters_string = f"filename = {tail2}, reslut pic num  = {testnum}, show pic num = {num}, <br> alpha = {alpha}, beta = {beta}, gamma = {gamma}, cutoff = {cutoff}"

    return parameters_string

def generate_html_content(root, subfolder_name, num):
    """
    生成HTML内容
    """
    image_names = get_image_names(root, subfolder_name, num)
    parameters = read_parameters(os.path.join(
        root, subfolder_name, 'saveData.txt'))

    str_content = f'<p>{parameters}</p>\n'
    img_content = '<p>'
    for image_name in image_names:
        img_content += f'<img src="{os.path.join(root, subfolder_name, image_name)}" />\n'
    img_content += '</p>\n'

    return str_content + img_content


html_content = """  
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
    <style>  
        p {  
            margin: 0;  
            font-weight: 300;  
            font-size: medium;  
        }  
        img {  
            width: 700px;  
        }  
    </style>  
</head>  
  
<body>  
"""

# 获取子文件夹名列表
subfolder_names = get_subfolder_names(root)

for subfolder_name in subfolder_names:
    html_content += generate_html_content(root, subfolder_name, num)

# 添加闭合标签
html_content += """  
</body>  
</html>  
"""

# 将html_content写入文件
output_file_path = os.path.join(root, 'visual_3pic.html')
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(html_content)
