# Reference: https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/sample_code.html
import torch
import collections  # 导入collections模块

def pytorch_params_to_txt(pth_file, output_file):
    # 加载.pth文件中的参数
    par_dict = torch.load(pth_file, map_location='cpu')
    
    # 打开输出文件
    with open(output_file, 'w') as f:
        # 定义一个递归函数来处理嵌套结构
        def process_items(items, prefix=''):
            for name, item in items:
                # 构造参数的完整名称
                full_name = f"{prefix}{name}"
                if isinstance(item, torch.Tensor):
                    # 如果是Tensor，将其名称和形状写入文件
                    shape_str = str(tuple(item.size()))
                    f.write(f"{full_name[7:]} {shape_str}\n")
                elif isinstance(item, (dict, collections.OrderedDict)):
                    # 如果是字典，递归处理
                    process_items(item.items(), prefix=f"{full_name}.")

        # 调用递归函数处理可能的嵌套字典
        process_items(par_dict.items())

# 指定.pth文件路径
# pth_path = "pretrained_models/pretrained_HQPs.pth"
pth_path = "pretrained_models/pretrained_RIDCP.pth"
        
# 指定输出文件路径
# output_txt_path = "pth_parameters.txt"
output_txt_path = "RIDCP_pth_parameters.txt"

# 调用函数
pytorch_params_to_txt(pth_path, output_txt_path)
print("Parameters have been written to", output_txt_path)


