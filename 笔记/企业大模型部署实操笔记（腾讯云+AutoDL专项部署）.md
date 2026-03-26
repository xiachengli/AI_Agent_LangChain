# 企业大模型部署实操笔记（腾讯云+AutoDL专项部署）

## 一、核心工具简单介绍

- Dify：一款开源的大模型应用开发平台，可快速搭建对话、问答等大模型应用，支持对接多种大模型（如Ollama、Xinference部署的模型），提供可视化界面，无需复杂编码即可完成应用配置与部署，本次部署用于提供大模型应用交互入口。

- AutoDL：一款云GPU算力平台，提供高性能、高性价比的GPU实例，预装多种适配大模型部署的镜像（如Pytorch+CUDA+Python），无需手动搭建复杂GPU环境，本次用于部署Ollama、Xinference及相关模型，支撑模型推理运行。

- Ollama：轻量级大模型部署工具，支持快速拉取、运行开源大模型（如Deepseek、Llama等），支持GPU加速，部署简单、占用资源适中，可通过命令行或接口调用模型，本次用于拉取并运行Deepseek对话模型。

- Xinference：开源的大模型推理部署工具，支持部署嵌入模型、重排序模型、对话模型等多种类型模型，兼容多种推理引擎（如vllm），可通过WebUI或接口管理模型，本次用于部署BGE-M3嵌入模型与BGE-Reranker-V2-M3重排序模型。

## 二、部署前准备（核心前提，决定部署效率与稳定性）

### 1. 明确部署目标与需求

- 核心需求：完成腾讯云Docker部署Dify、AutoDL部署Ollama与Xinference，实现Ollama拉取Deepseek模型、Xinference部署嵌入模型与重排序模型，搭建完整的大模型应用支撑环境

- 性能要求：腾讯云服务器需满足Docker运行基础配置（CPU≥4核、内存≥8GB、SSD≥50GB）；AutoDL实例需匹配模型部署需求（优先选择NVIDIA GPU，显存≥16GB，支持CUDA，推荐Pytorch2.3.0+Python3.10+CUDA12.1镜像）

- 合规要求：确认数据隐私（是否涉及敏感数据，需符合《数据安全法》《个人信息保护法》），腾讯云与AutoDL实例需配置安全组，开放必要端口

### 2. 技术与资源评估

- 硬件资源：腾讯云服务器（推荐轻量应用服务器或云服务器ECS，系统选择Ubuntu 20.04/22.04）；AutoDL实例（GPU优先NVIDIA系列，显存≥16GB，用于部署Ollama与Xinference，支撑模型运行）

- 软件环境：腾讯云端（Ubuntu 20.04/22.04、Docker、Docker Compose）；AutoDL端（Ubuntu系统、Ollama、Xinference、vllm引擎、sentence-transformers，Git-LFS）

### 3. 模型选型（关键步骤，贴合本次部署需求）

- 核心模型：对话模型（Deepseek，推荐deepseek-r1:7b版本，显存占用适中，适配AutoDL实例，支持断点续传下载）；嵌入模型（推荐BGE-M3，适配Xinference，量化后性能更优）；重排序模型（推荐BGE-Reranker-V2-M3，轻量高效，适配Xinference部署）

- 模型来源：Deepseek通过Ollama官方仓库拉取；嵌入模型、重排序模型通过ModelScope仓库下载（配置XINFERENCE_MODEL_SRC=modelscope可提升下载速度）

- 模型格式：优先选择量化模型（INT4/INT8），降低显存占用，提升推理速度，适配AutoDL实例硬件资源，减少部署卡顿

## 三、核心部署步骤（实操重点，按腾讯云→AutoDL顺序执行，避免遗漏）

### 步骤1：腾讯云服务器环境搭建与Docker部署

1. 腾讯云实例配置：登录腾讯云控制台，云服务器CVM创建Ubuntu Server 24.04 LTS 64位系统实例，开放端口（80、443、5001，分别用于基础访问、HTTPS、Dify服务），重置实例密码并通过SSH连接（命令：ssh 用户名@实例公网IP）

2. Xshell免费版的官方下载页面是：[https://www.xshell.com/zh/free-for-home-school/](https://www.xshell.com/zh/free-for-home-school/)

3. 安装Docker与Docker Compose：
`# 1：更新系统软件包
sudo apt update && sudo apt upgrade -y
# 2：安装Docker依赖
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
# 使用阿里云Docker镜像（国内腾讯云服务器优先选择，解决官方源访问慢问题）
# 阿里云镜像优势：国内访问延迟低、下载速度快，稳定性高，无需科学上网
# 添加阿里云Docker GPG密钥（当前命令）
# 核心：将文本格式密钥转为二进制格式（--dearmor作用），保存到系统推荐的密钥目录
# 3：导入阿里云Docker GPG密钥
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# 添加阿里云Docker软件源（详解如下）
# 完整命令拆解：echo "软件源配置内容" | sudo tee 软件源文件 > /dev/null
# 4：配置阿里云Docker软件源
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu noble stable" | sudo tee /etc/apt/sources.list.d/docker.list
# 5：安装Docker
sudo apt install -y docker-ce docker-ce-cli containerd.io
# 6：启动Docker并设置开机自启
sudo systemctl start docker && sudo systemctl enable docker`

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/ec857f7c98b845c8be2dc4611139a8b7.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=cDc8%2BimofjllFumBNEwwGA%2F6dNs%3D&resource_key=bd34762c-6f9e-4c94-8aba-20e75e426c0f&resource_key=bd34762c-6f9e-4c94-8aba-20e75e426c0f)

`# 7：安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose`

`执行失败，添加镜像源配置`

`sudo vi /etc/docker/daemon.json`

`添加如下（已检查优化，适配腾讯云Ubuntu环境）：`

{

"registry-mirrors":[

"https://docker.mirrors.ustc.edu.cn",  # 中科大镜像，稳定且速度快，优先推荐

"https://hub-mirror.c.163.com",      # 网易镜像，国内节点多，兼容性好

"https://5tqw56kt.mirror.aliyuncs.com", # 阿里云镜像，适配腾讯云网络，延迟低

"https://docker.m.daocloud.io",      #  DaoCloud镜像，备用稳定

"https://docker.nju.edu.cn"          # 南京大学镜像，备用，适合华东地区

]

}

`添加后执行 sudo systemctl daemon-reload && sudo systemctl restart docker 使配置生效`

`# 8：赋予Docker Compose执行权限
sudo chmod +x /usr/local/bin/docker-compose
# 9：验证Docker与Docker Compose安装
docker --version && docker-compose --version`

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/22e87dbd60b049c69f3bc3b3dffe923c.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=ci8i1HHRZVg2oO10KBByeTGp2lA%3D&resource_key=e7b2a6aa-37db-4566-9013-53d4df13c68e&resource_key=e7b2a6aa-37db-4566-9013-53d4df13c68e)

1. Docker中部署Dify：
       `# 10：创建Dify部署目录并进入
mkdir -p ~/dify && cd ~/dify
` `# 11：下载Dify`

2. `sudo git clone ` `https://gitee.com/shkstart/dify.git`

3. `# 12：启动Dify服务（后台运行）`

    `cd /opt/dify/docker` `
    ` `sudo docker-compose up -d`

`# 13：验证Dify部署（查看容器运行状态）
` `sudo docker ps | grep dify`

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/32684616a525447fa7e3bae6b8c6793f.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=v8ZMjChT2qkwRBMqbQcL5nN1gnw%3D&resource_key=2ebee89a-de38-4f76-b4ad-0cb6f09c9424&resource_key=2ebee89a-de38-4f76-b4ad-0cb6f09c9424)

`# 14：访问Dify（浏览器输入对应地址，完成初始化）
` `# 浏览器输入 http://腾讯云实例公网IP:`80

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/24ae5ba30a494a8aa4feba18b9b903f3.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=hsQNEWfiHH6DniWluScKNJMhwac%3D&resource_key=e98588a0-f00e-4eaa-8bd4-9efc6fd67276&resource_key=e98588a0-f00e-4eaa-8bd4-9efc6fd67276)

查看日志

docker compose logs -f plugin_daemon

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/68fe436e54d2476b9313d84d196645a4.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=JXNXQZodrAE75J624ZUux6WthWA%3D&resource_key=ae20f9d2-9f2a-42e9-8481-0f4126e8600b&resource_key=ae20f9d2-9f2a-42e9-8481-0f4126e8600b)

营养师助手：

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/7a7bc2b1eb7c42bd80b875be27887736.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=IOIcoNd3zncerk3c%2BHKCIVixkvM%3D&resource_key=36c16c61-5b3f-4354-9073-1da479f43de7&resource_key=36c16c61-5b3f-4354-9073-1da479f43de7)

### 步骤2：AutoDL实例环境搭建（部署Ollama与Xinference）

1. AutoDL实例配置：登录AutoDL控制台，选择适配镜像（Pytorch2.1.2+Python3.10+CUDA11.8），创建实例，记录实例公网IP与SSH端口，通过SSH连接实例（命令：ssh root@实例公网IP -p 实例SSH端口）

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/32abe098605c442da27c9594216e78d9.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=pxgqbdkVMRvTkqd%2FlIkVZzBlvKo%3D&resource_key=a6ab6c8b-14a5-445a-98b4-d8d668099d9d&resource_key=a6ab6c8b-14a5-445a-98b4-d8d668099d9d)

### 步骤3：AutoDL部署Ollama并拉取Deepseek模型

学术资源加速：

source /etc/network_turbo

1. 拉取并启动Ollama：
`# 15：安装Ollama（AutoDL实例直接执行，支持GPU加速）`

2. `mkdir ./dify`

3. `cd dify` `
` `curl -fsSL https://ollama.com/install.sh | `sh

4. `# 16：启动Ollama服务（后台运行）
` `OLLAMA_HOST=` `127.0.0.1:6006` ` ollama serve &`

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/f3d46615b0de419686de2ce287b04cb5.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=2%2Ff5oxrLlJSYgr4WRZcWirEu3Y4%3D&resource_key=fdbdce05-76d2-4663-8d81-1560a307e00f&resource_key=fdbdce05-76d2-4663-8d81-1560a307e00f)

`# 17：验证Ollama启动状态
` `ps aux | grep ollama`

1. 拉取并测试Deepseek模型（无需进入Docker容器）：
        `# 18：拉取Deepseek模型（7B版本，显存占用约8GB，适合AutoDL实例）
ollama pull deepseek-r1:7b
# 可选19：拉取14B版本（需显存≥16GB）
# 20：测试模型（启动模型并进行简单交互）
ollama run deepseek-r1:7b
# 21：退出模型交互
# 输入/exit退出交互`

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/7507d52ab5a24675b3db3bc813bfe67b.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=zHSigZpKX0EMYVaOYHzrgxwIj8w%3D&resource_key=0e962af0-f881-4df5-b8cd-a55131253ce9&resource_key=0e962af0-f881-4df5-b8cd-a55131253ce9)

1. 验证Ollama服务：在AutoDL实例中执行curl命令，验证服务可用性：
`# 22：验证Ollama服务可用性
curl http://localhost:11434/api/generate -d '{ "model":"deepseek-r1:7b","prompt":"介绍大模型部署流程","stream":false}'`

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/53b93cb0a87c406ba116dae55562eecc.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=gO9wraD%2F0W4vI0N48CuIZmZXiW0%3D&resource_key=53f38843-2ab7-42cc-af1e-60cced52a7fd&resource_key=53f38843-2ab7-42cc-af1e-60cced52a7fd)

1. Dify打通ollama隧道并使用deepseek

    ![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/0546c23c9ef94a748e0c7be44934107a.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=yDL%2FCvrgDy0gTywJhCzbJxfxc7s%3D&resource_key=9549cfed-7be9-4dd6-b792-0cee3270ad27&resource_key=9549cfed-7be9-4dd6-b792-0cee3270ad27)

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/982e4c5b72434bd78d7a4662c3766444.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=TD1IxavZWPkc1AAU5KR7CVB2BGw%3D&resource_key=15f367bf-07ba-4e6a-bb68-d09c822c03ad&resource_key=15f367bf-07ba-4e6a-bb68-d09c822c03ad)

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/7d73689941b24d6cade9ad925e3e8aff.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=%2F5J4XEDX3HO24v7zw7kCst6ON3s%3D&resource_key=1bb2dc29-3b50-4bc1-8ddf-0838eb77f0e8&resource_key=1bb2dc29-3b50-4bc1-8ddf-0838eb77f0e8)



![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/65d7133568be465ea0b087485146c3ff.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=oQcXUUawaB%2FpiJpY8WFA2Dj%2BmPw%3D&resource_key=b2500b4b-44cc-4c93-b7dc-fe3603dd8f54&resource_key=b2500b4b-44cc-4c93-b7dc-fe3603dd8f54)

### 步骤4：AutoDL部署Xinference并部署嵌入模型、重排序模型

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/e0ad1d6af9634cc685b873893a1540f9.png~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1774541682&x-signature=3QTmgK0stMEpE03VFUATJGM0sTw%3D&resource_key=5c35c32c-3361-4332-b5b6-8605d9913d6b&resource_key=5c35c32c-3361-4332-b5b6-8605d9913d6b)

1. 建立SSH隧道（用于本地访问Xinference WebUI）：
        `# 32：本地终端执行（替换为AutoDL实例信息，建立SSH隧道）
ssh -CNg -L 9997:127.0.0.1:9997 root@AutoDL实例公网IP -p AutoDL实例SSH端口
# 执行后输入实例密码，保持终端运行，即可通过本地浏览器访问 http://localhost:9997`

2. 部署嵌入模型（BGE-M3）：
      

3. 本地浏览器访问http://localhost:9997，进入Xinference WebUI

4. 点击左上角“Launch Model”，选择“EMBEDDING MODELS”，搜索“bge-m3”

5. 配置模型：选择vllm引擎，保持默认参数，点击启动按钮，等待模型下载并部署完成（可在AutoDL实例日志中查看部署进度）

6. 测试嵌入模型：
          `# 33：测试BGE-M3嵌入模型
from langchain_community.embeddings import XinferenceEmbeddings
server_url="http://localhost:9997/"
model_uid = "bge-m3"  # 需与WebUI中启动后的Model UID一致
embed = XinferenceEmbeddings(server_url=server_url, model_uid=model_uid)
# 测试嵌入生成
print(embed.embed_query("大模型部署"))`

部署重排序模型（BGE-Reranker-V2-M3）：
      

1. 在Xinference WebUI中，点击“Launch Model”，选择“RERANK MODELS”，搜索“bge-reranker-v2-m3”

2. 配置模型：选择vllm引擎，保持默认参数，点击启动按钮，等待部署完成

3. 验证部署：在WebUI“Running Models”中查看模型状态，确认状态为“Running”即可正常使用

### 步骤5：接口测试与联调（确保各服务正常通信）

1. Dify服务测试：浏览器访问http://腾讯云实例公网IP:5001，登录管理员账号，创建应用，测试对话功能，确认Dify服务正常运行

2. Ollama与Deepseek测试：在AutoDL实例中再次执行模型交互命令，或通过curl调用接口，确认Deepseek模型响应正常；若需与Dify联调，需在Dify中配置Ollama接口（地址：http://腾讯云内网IP:6006）

3. Xinference模型测试：分别调用嵌入模型与重排序模型接口，验证模型输出正常；若需与Dify联调，在Dify中配置Xinference接口（地址：http://AutoDL实例公网IP:9997）

4. 异常排查：若服务启动失败，查看对应日志（Dify日志：docker logs 容器ID；Xinference日志：cat xinference.log；Ollama日志：docker logs ollama）；若端口无法访问，检查腾讯云与AutoDL安全组配置，开放对应端口

### 步骤6：监控与运维（长期稳定运行的关键）

- 监控指标：腾讯云服务器（CPU/内存/磁盘使用率、Docker容器运行状态，Docker基于阿里云镜像部署）；AutoDL实例（GPU利用率、显存占用、Ollama与Xinference服务状态）；各模型接口响应时间、错误率

- 监控工具：腾讯云控制台（查看服务器状态）；AutoDL控制台（查看GPU使用情况）；可额外部署Prometheus + Grafana，实现可视化监控与告警

- 日常运维：定期备份模型文件（Ollama模型：本地存储路径/root/.ollama；Xinference模型：XINFERENCE_HOME目录）；更新阿里云Docker镜像、依赖包与模型版本；检查SSH隧道连接状态，若中断需重新执行隧道命令

- 日志管理：收集Dify、Ollama、Xinference日志，用于问题排查；定期清理日志，避免占用过多存储空间

## 四、关键注意事项（避坑重点，必看）

- 数据安全：部署过程中，避免敏感数据泄露，模型训练/推理用的数据需脱敏处理；若使用开源模型，确认模型授权协议，避免侵权；腾讯云与AutoDL实例需设置复杂密码，关闭不必要的端口

- 硬件成本：AutoDL实例可根据模型需求选择按需计费模式，降低前期投入；腾讯云服务器可根据Dify服务负载，选择合适的配置，避免资源浪费

- 版本兼容：严格匹配AutoDL镜像、CUDA版本、Xinference引擎版本，避免出现“引擎安装失败”“模型部署报错”等问题；Ollama版本需支持Deepseek模型拉取，建议使用最新稳定版

- 合规风险：金融、医疗等行业，需确保模型部署符合行业监管要求，必要时进行合规审计；模型下载需通过正规渠道，避免下载恶意篡改的模型文件

- 灾备方案：定期备份腾讯云Dify配置与AutoDL模型文件；AutoDL实例若出现故障，可快速重新部署环境，通过备份文件恢复模型与配置

## 五、常见问题排查（实操必备）

1. 问题1：Ollama拉取Deepseek模型中断
        解决：重新执行ollama pull deepseek-r1:7b命令（18），Ollama支持断点续传；若下载速度过慢，检查AutoDL实例网络，或更换模型版本（7B版本体积更小，下载更快）
      

2. 问题2：Xinference启动失败，提示“引擎加载失败”
        解决：优先使用vllm引擎，重新执行pip install "xinference(vllm)"命令（25）；检查CUDA版本与vllm引擎兼容性，确保AutoDL实例GPU支持CUDA；当前已默认使用阿里云Docker镜像，无需额外替换，若仍出现Docker安装失败，可重新执行步骤1中阿里云镜像配置相关命令（3、4）
      

3. 问题3：本地无法访问Xinference WebUI
        解决：检查SSH隧道是否正常运行（本地终端未关闭，对应32）；确认AutoDL实例9997端口已开放；检查隧道命令中的IP、端口是否正确
      

4. 问题4：Dify无法连接Ollama/Xinference服务
       解决：检查AutoDL实例安全组是否开放11434（Ollama）、9997（Xinference）端口；确认Dify中配置的接口地址正确（AutoDL实例公网IP+对应端口）；检查AutoDL实例中对应服务是否正常运行（Ollama对应16、17，Xinference对应30、31）

5. 问题5：Xinference部署模型时，提示“模型下载失败”
        解决：确认已配置export XINFERENCE_MODEL_SRC=modelscope（28）；检查Git-LFS是否安装成功；重新启动Xinference服务（30），再次尝试部署
      

## 六、部署总结

本次专项部署核心逻辑：腾讯云搭建Docker环境部署Dify，提供大模型应用交互入口；AutoDL实例部署Ollama与Xinference，分别承载Deepseek对话模型、BGE-M3嵌入模型与BGE-Reranker-V2-M3重排序模型，形成完整的大模型部署链路。部署重点是做好环境适配（腾讯云与AutoDL实例配置）、端口开放、模型拉取与服务联调，同时关注版本兼容与数据安全。实操中需按步骤执行，重点排查端口、日志与服务状态，确保各环节衔接顺畅，保障服务长期稳定运行。
> （注：文档部分内容可能由 AI 生成）