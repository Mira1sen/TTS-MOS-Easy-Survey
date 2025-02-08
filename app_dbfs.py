#demo FYI
import gradio as gr
import os
import random
import torch
import numpy as np
import torchaudio.transforms as T
import torchaudio

def preprocess_audio_torch(waveform, original_sample_rate, target_sample_rate=24000):
    # Resample the waveform to the target sample rate (16kHz)
    resample = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    waveform = resample(waveform)

    # Ensure the audio is mono (downmix stereo if needed)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert waveform to NumPy array for further processing
    waveform_np = waveform.numpy().flatten()

    # Convert to 16-bit PCM (like setting sample width to 2 bytes)
    # In PyTorch, we already work with float32 by default
    waveform_np = waveform_np.astype(np.float32)

    # Calculate dBFS (decibel relative to full scale) of the audio
    rms = np.sqrt(np.mean(waveform_np ** 2))
    dBFS = 20 * np.log10(rms) if rms > 0 else -float('inf')

    # Calculate the gain to be applied (target dBFS is -20)
    target_dBFS = -30
    gain = target_dBFS - dBFS
    # logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Apply gain, limiting it between -3 and 3 dB
    gain = min(max(gain, -20), 20)
    waveform_np = waveform_np * (10 ** (gain / 20))

    # Normalize waveform (max absolute amplitude should be 1.0)
    max_amplitude = np.max(np.abs(waveform_np))
    if max_amplitude > 0:
        waveform_np /= max_amplitude

    # logger.debug(f"waveform shape: {waveform_np.shape}")
    # logger.debug(f"waveform dtype: {waveform_np.dtype}")

    return waveform_np, target_sample_rate

class MOSApp:
    MOS = {
        1: "1-Bad", 1.5: "1.5", 2: "2-Poor", 2.5: "2.5", 3: "3-Fair",
        3.5: "3.5", 4: "4-Good", 4.5: "4.5", 5: "5-Excellent"
    }

    def __init__(self):
        # 在初始化时清理旧的临时文件
        self.cleanup_temp_files()
        
        random.seed(10)
        self.prompt_floder = "prompts"
        
        # 动态检测模型文件夹
        models_results_path = './models_results'
        model_folders = [os.path.join(models_results_path, folder) 
                        for folder in os.listdir(models_results_path) 
                        if os.path.isdir(os.path.join(models_results_path, folder))]
        self.model_folders = sorted(model_folders)
        
        # 获取每个模型文件夹下的音频文件
        self.model_files = {}
        for folder in self.model_folders:
            self.model_files[folder] = sorted([os.path.join(folder, file) 
                                               for file in os.listdir(folder) 
                                               if file.endswith("wav") or file.endswith("mp3")
            ])
        
        # 获取prompt文件夹下的音频文件
        self.prompt_files = sorted([os.path.join(self.prompt_floder, file) for file in os.listdir(self.prompt_floder)])
        # 确保所有模型文件夹的音频数量与prompt一致
        min_length = min(len(self.prompt_files), *(len(files) for files in self.model_files.values()))
        self.prompt_files = self.prompt_files[:min_length]
        for folder in self.model_folders:
            self.model_files[folder] = self.model_files[folder][:min_length]
        
        # 为每个音频组创建随机顺序
        self.audio_order = []
        for i in range(len(self.prompt_files)):
            models = [(self.model_files[folder][i], folder, idx) for idx, folder in enumerate(self.model_folders, start=1)]
            random.shuffle(models)
            self.audio_order.append(models)

        print(self.audio_order)

        # 添加已使用ID的集合
        self.used_ids = set()
        # 如果results.csv存在，读取所有已使用的ID
        if os.path.exists('results.csv'):
            try:
                with open('results.csv', 'r') as f:
                    # 跳过表头
                    next(f, None)
                    for line in f:
                        if line.strip():
                            # 获取每行的第一个字段（ID）
                            used_id = line.split(',')[0]
                            self.used_ids.add(used_id)
            except Exception as e:
                print(f"读取results.csv时出错: {e}")

        # 添加音频处理的设置
        self.target_sample_rate = 24000
        self.process_audio_files()  # 添加这行来预处理所有音频

    def initialize_state(self):
        return {
            "index": 0,
            "selected_MOS": {folder: [] for folder in self.model_folders},  # 各模型的评分
            "tester_id": "",
            "data_store": {}
        }

    def submit_options(self, state, *options):
        if not state["tester_id"]:
            return (
                *([None] * (1 + len(self.model_folders))),
                "请先输入您的测试者ID",
                *([0.5] * len(self.model_folders)),
                state
            )
        
        if state["index"] >= len(self.prompt_files):
            return (
                *([None] * (1 + len(self.model_folders))),
                "## 测评已经结束！感谢您的反馈！\n## ",
                *([0.5] * len(self.model_folders)),
                state
            )
        
        # 检查所有评分是否都已选择
        if 0.5 in options:
            current_audios = self.audio_order[state["index"]]
            return (
                *(audio[0] for audio in current_audios),  # 各模型的音频
                self.prompt_files[state["index"]],
                "#### 无效提交！请为所有音频选择评分后再提交",
                *options,
                state
            )

        # 根据随机顺序记录评分
        current_order = self.audio_order[state["index"]]
        for i, (_, folder, _) in enumerate(current_order):
            state["selected_MOS"][folder].append(options[i])

        state["index"] += 1
        self.save_state(state)  # 保存更新后的状态
        
        if state["index"] < len(self.prompt_files):
            next_audios = self.audio_order[state["index"]]
            return (
                *(self.process_audio(audio[0]) for audio in next_audios),  # 使用处理后的音频
                self.process_audio(self.prompt_files[state["index"]]),
                f"#### 您正在评价第 {state['index']+1} 组音频，共 {str(len(self.prompt_files))} 组。提交后请向上滚动收听新的音频",
                *([0.5] * len(self.model_folders)),
                state
            )
        else:
            # 保存所有评分结果为CSV格式
            file_exists = os.path.isfile('results.csv')
            self.used_ids.add(state["tester_id"])
            with open('results.csv', 'a') as f:
                if not file_exists:
                    header = "id,model," + ",".join([f"MOS{i+1}" for i in range(len(self.prompt_files))])
                    f.write(header + "\n")
                
                tester_id = state["tester_id"]
                for folder in self.model_folders:
                    scores = ",".join(map(str, state["selected_MOS"][folder]))
                    f.write(f"{tester_id},{folder},{scores}\n")
            
            return (
                *([None] * (1 + len(self.model_folders))),
                "## 感谢您的反馈！您的测评数据已保存",
                *([0.5] * len(self.model_folders)),
                state
            )

    def set_tester_id(self, id, state):
        if not id:
            return (
                "## 请输入有效的ID！", 
                state, 
                *([None] * (1 + len(self.model_folders))),
                ""
            )
        
        # 检查是否有保存的状态
        saved_state = self.load_state(id)
        if id in self.used_ids:
            return (
                "## 该ID数据已记录，请使用新的ID！", 
                state, 
                *([None] * (1 + len(self.model_folders))),
                ""
            )
        
        if saved_state:
            state.update(saved_state)
            current_audios = self.audio_order[state["index"]]
            return (
                f"## 您的ID: {state['tester_id']}（已恢复之前的进度）", 
                state, 
                *(self.process_audio(audio[0]) for audio in current_audios),
                self.process_audio(self.prompt_files[state["index"]]),
                f"#### 您正在评价第 {state['index']+1} 组音频，共 {str(len(self.prompt_files))} 个。提交后请向上滚动收听新的音频"
            )
        
        state["tester_id"] = id
        for folder in self.model_folders:
            state["selected_MOS"][folder] = []
        state["index"] = 0
        
        # 保存初始状态
        self.save_state(state)
        
        first_audios = self.audio_order[0]
        return (
            f"## 您的ID: {state['tester_id']}", 
            state, 
            *(self.process_audio(audio[0]) for audio in first_audios),
            self.process_audio(self.prompt_files[0]),
            f"#### 您正在评价第 {state['index']+1} 组音频，共 {str(len(self.prompt_files))} 个。提交后请向上滚动收听新的音频"
        )

    def create_interface(self):
        with gr.Blocks() as demo:
            state = gr.State(self.initialize_state())

            gr.Markdown("# TTS MOS Easy Survey")

            gr.Markdown("## 首先请输入您的测试者ID")
            
            with gr.Row():
                tester_id_input = gr.Textbox(label="输入测试者ID")
                set_id_button = gr.Button("确认ID")
                id_display = gr.Markdown()
            
            gr.Markdown("------")
            gr.Markdown("## 请根据以下标准为音频打分")
            with gr.Row():
                with gr.Column(scale=1):
                    score_description = gr.Markdown("""
                        ### 请从自然度、清晰度和发音准确度方面评价语音的整体质量
                        | 分数 | 自然度/人声相似度 | 机器音特征 |
                        |-------|---------------------|-------------|
                        | 5 优秀 | 完全自然的语音 | 无法察觉机器音特征 |
                        | 4 良好 | 大部分自然的语音 | 可以察觉但不影响听感 |
                        | 3 一般 | 自然与不自然程度相当 | 明显可察觉且略有影响 |
                        | 2 较差 | 大部分不自然的语音 | 令人不适但尚可接受 |
                        | 1 很差 | 完全不自然的语音 | 非常明显且无法接受 |
                        """)
                with gr.Column(scale=2):
                    gr.Markdown("### 评分参考示例")
                    with gr.Row():
                        gr.Audio("assets/example_mos5.wav", label="5分示例", scale=1)
                        gr.Audio("assets/example_mos3.wav", label="3分示例", scale=1)
                        gr.Audio("assets/example_mos1.wav", label="1分示例", scale=1)
            gr.Markdown("------")
            gr.Markdown("## 请仔细听以下prompt音频：")
            with gr.Row():
                prompt_audio = gr.Audio(
                    value=None,
                    type='filepath',
                    label="Prompt音频"
                )

            gr.Markdown("------")
            gr.Markdown("## 请仔细听以下不同模型的音频：")
            
            audio_elements = []
            option_elements = []
            label_elements = []
            for i, folder in enumerate(self.model_folders):
                gr.Markdown(f"### audio {i+1}：")
                audio = gr.Audio(None, type='filepath')
                option = gr.Slider(minimum=0.5, maximum=5, step=0.5,
                                  value=0.5, container=False, interactive=True)
                label = gr.HTML(self.get_slider_labels_html())
                audio_elements.append(audio)
                option_elements.append(option)  # 保存滑块实例
            
            with gr.Row():
                submit = gr.Button("提交")
            current_file = gr.Markdown("#### 请先输入ID再开始评测")

            set_id_button.click(
                self.set_tester_id, 
                inputs=[tester_id_input, state], 
                outputs=[id_display, state, *audio_elements, prompt_audio, current_file]
            )
            submit.click(
                self.submit_options, 
                inputs=[state, *option_elements], 
                outputs=[*audio_elements, prompt_audio, current_file, *option_elements, state]
            )

        return demo

    @staticmethod
    def get_slider_labels_html():
        return """
        <style>
            .slider-labels {
                display: flex;
                justify-content: space-between;
                margin-top: -10px;
                font-size: 12px;
            }
            .slider-labels div {
                text-align: center;
                width: 5%;
            }
            .slider-labels div:first-child {
                text-align: left;
            }
            .slider-labels div:last-child {
                text-align: right;
            }
        </style>
        <div class="slider-labels">
            <div>请选择</div>
            <div>1 很差</div>
            <div>1.5</div>
            <div>2 较差</div>
            <div>2.5</div>
            <div>3 一般</div>
            <div>3.5</div>
            <div>4 良好</div>
            <div>4.5</div>
            <div>5 优秀</div>
        </div>
        """

    def save_state(self, state):
        """保存当前状态到本地文件"""
        state_file = f"states/{state['tester_id']}.json"
        os.makedirs('states', exist_ok=True)
        
        with open(state_file, 'w') as f:
            import json
            json.dump({
                'index': state['index'],
                'selected_MOS': state['selected_MOS'],
                'tester_id': state['tester_id']
            }, f)

    def load_state(self, tester_id):
        """从本地文件加载状态"""
        state_file = f"states/{tester_id}.json"
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                import json
                saved_state = json.load(f)
                return saved_state
        return None

    def process_audio(self, audio_path):
        """处理单个音频文件"""
        try:
            # 创建基于原始路径的唯一标识符
            unique_id = audio_path.replace('/', '_').replace('\\', '_')
            
            # 将处理后的音频保存到临时目录
            os.makedirs('temp_audio', exist_ok=True)
            temp_path = os.path.join('temp_audio', f"processed_{unique_id}")
            
            # 如果已经处理过这个文件，直接返回处理后的路径
            if os.path.exists(temp_path):
                return temp_path
            
            waveform, sample_rate = torchaudio.load(audio_path)
            processed_waveform, _ = preprocess_audio_torch(waveform, sample_rate, self.target_sample_rate)
            
            # 保存为WAV文件
            processed_waveform = torch.from_numpy(processed_waveform).unsqueeze(0)
            torchaudio.save(temp_path, processed_waveform, self.target_sample_rate)
            
            return temp_path
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return audio_path
    
    def process_audio_files(self):
        """处理所有音频文件"""
        # 处理prompt文件
        self.processed_prompt_files = [self.process_audio(f) for f in self.prompt_files]
        
        # 处理模型音频文件
        self.processed_model_files = {}
        for folder in self.model_folders:
            self.processed_model_files[folder] = [
                self.process_audio(f) for f in self.model_files[folder]
            ]

    def cleanup_temp_files(self):
        """清理临时音频文件"""
        import shutil
        if os.path.exists('temp_audio'):
            shutil.rmtree('temp_audio')

if __name__ == "__main__":
    app = MOSApp()
    demo = app.create_interface()
    demo.launch(server_name="0.0.0.0", server_port=8565, show_error=True)
