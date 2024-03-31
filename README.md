# Healthcare-Agent 医疗保健智能体

### 1. 项目介绍 
GenAI赋能，解读用户的健康密码，根据个性化的数据，生成易懂化的解读，为家庭健康决策人提供全寿命周期的健康管理大语言模型智能体，成为陪伴用户一生的健康伴侣。
   
### 2. HealthcareAgent 医疗保健智能体 2B 主要功能：  
  - 智能问答
  - CVD(心血管）慢病管理
  - 数据分析（报告解读及对比分析）


### 3. 模型工具
[InternLM2](https://github.com/InternLM/InternLM)    [Lagent](https://github.com/InternLM/lagent)  
[开源代码](https://open-compass.github.io/T-Eval/)  
[项目主页](open-compass.github.io/T-Eval)  
[论文](arxiv.org/abs/2312.14033)  

      
### 4. Lagent2 智能体工具调用  
#### 4.1 环境准备 
选择和第一个 `InternLM` 一样的镜像环境，运行以下命令安装依赖，如果上一个 `InternLM2-Chat-20B` 已经配置好环境不需要重复安装.

       # 升级pip
         python -m pip install --upgrade pip

         pip install modelscope==1.9.5
         pip install transformers==4.35.2
         pip install streamlit==1.24.0
         pip install sentencepiece==0.1.99
         pip install accelerate==0.24.1
        
#### 4.2 模型下载  
InternStudio 平台的 `share` 目录下已经为我们准备了全系列的 `InternLM` 模型，所以我们可以直接复制即可。使用如下命令复制： 
         
         mkdir -p /root/data/model/Shanghai_AI_Laboratory
         cp -r /root/share/model_repos/internlm2-chat-7b /root/data/model/Shanghai_AI_Laboratory   
  
#### 4.3 Lagent2 安装  
首先切换路径到 `/root/code` 克隆 `lagent` 仓库，并通过 `pip install -e .` 源码安装 `Lagent2`  

        cd /root/code  
        pip install lagent streamlit  -U 
               
#### 4.4 修改代码  
由于代码修改的地方比较多，大家直接将 `/root/code/lagent/examples/react_web_demo.py` 内容替换为以下代码  

[lagent/examples/internlm2_agent_web_demo.py](https://github.com/InternLM/lagent/blob/main/examples/internlm2_agent_web_demo.py)    


#### 4.5 运行一个智能体的网页样例  
