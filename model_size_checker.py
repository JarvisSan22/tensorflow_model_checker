from tensorflow.keras.utils import plot_model 
from tensorflow.keras import backend as K 
import numpy as np 
import subprocess

def get_model_memory_usage(batch_size, model): 
     
    features_mem = 0 # Initialize memory for features.  
    float_bytes = 4.0 #Multiplication factor as all values we store would be float32. 
     
    for layer in model.layers: 
        out_shape = layer.output_shape 
         
        if type(out_shape) is list:   #e.g. input layer which is a list 
            out_shape = out_shape[0] 
        else: 
            out_shape = [out_shape[1], out_shape[2], out_shape[3]] 
             
        #Multiply all shapes to get the total number per layer.     
        single_layer_mem = 1  
        for s in out_shape: 
            if s is None: 
                continue 
            single_layer_mem *= s 
         
        single_layer_mem_float = single_layer_mem * float_bytes #Multiply by 4 bytes (float) 
        single_layer_mem_MB = single_layer_mem_float/(1024**2)  #Convert to MB 
         
        print("Memory for", out_shape, " layer in MB is:", single_layer_mem_MB) 
        features_mem += single_layer_mem_MB  #Add to total feature memory count 
# Calculate Parameter memory 
    trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights]) 
    non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights]) 
    parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes)/(1024**2) 
    print("_________________________________________") 
    print("Memory for features in MB is:", features_mem*batch_size) 
    print("Memory for parameters in MB is: %.2f" %parameter_mem_MB) 
    total_memory_MB = (batch_size * features_mem) + parameter_mem_MB  #Same number of parameters. independent of batch size 
    total_memory_GB = total_memory_MB/1024 
    print("Minimum memory required to work with this model is: %.2f" %total_memory_GB, "GB") 
    return total_memory_GB

default_properies = (
  "timestamp",
  "gpu_name",
  "index",
  "memory.total",
  "memory.used",
  "memory.free",
  "utilization.gpu",
  "utilization.memory",
)

def get_gpu_properties( cmd_path="nvidia-smi",
  target_properties=default_properies,
  noheader=True,
  nounits=True
  ):
  """
  CUDA GPUのプロパティ情報取得
 
  Parameters
  ----------
  cmd_path : str
    コマンドラインから"nvidia-smi"を実行する際のパス
  target_properties : obj
    取得するプロパティ情報
    プロパティ情報の詳細は"nvidia-smi --help-query-gpu"で取得可能
  noheader : bool
    skip the first line with column headers
  nounits : bool
    don't print units for numerical values
 
  Returns
  -------
  gpu_properties : list
    gpuごとのproperty情報
    参考
    https://www.12-technology.com/2022/01/pythongpu.html
  """
    
  # formatオプション定義
  format_option = "--format=csv"
  if noheader:
      format_option += ",noheader"
  if nounits:
      format_option += ",nounits"
 
  # コマンド生成
  cmd = '%s --query-gpu=%s %s' % (cmd_path, ','.join(target_properties), format_option)
 
  # サブプロセスでコマンド実行
  cmd_res = subprocess.check_output(cmd, shell=True)
    
  # コマンド実行結果をオブジェクトに変換
  gpu_lines = cmd_res.decode().split('\n')
  # リストの最後の要素に空行が入るため除去
  gpu_lines = [ line.strip() for line in gpu_lines if line.strip() != '' ]
 
  # ", "ごとにプロパティ情報が入っているのでdictにして格納
  gpu_properties = [ { k: v for k, v in zip(target_properties, line.split(', ')) } for line in gpu_lines ]
 
  return gpu_properties


def Check_runability(batch_size, model):
    gpu_info=get_gpu_properties()[0]
    model_size_GB=get_model_memory_usage(batch_size, model)
    print("model size GB :",model_size_GB)
    gpu_ram_GB=gpu_info["memory.total"]/1024
    print("GPU RAM GB:",gpu_ram_GB)
    if model_size_GB>gpu_ram_GB:
        print("Model too large, lower batch size, or change input data size")
        return False
    else:
        print("model can run")
        return True
