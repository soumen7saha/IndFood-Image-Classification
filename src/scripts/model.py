import os
import torch
from pydantic import BaseModel
from PIL import Image
import numpy as np
from torchvision import models, transforms
from torch import nn
import onnxruntime as ort

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
classes = ['aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_paratha',
 'aloo_shimla_mirch', 'aloo_tikki', 'amritsari_kulcha', 'anda_curry',
 'ariselu', 'balushahi', 'banana_chips', 'bandar_laddu', 'basundi',
 'besan_laddu', 'bhindi_masala', 'biryani', 'boondi', 'boondi_laddu',
 'butter_chicken', 'chaas', 'chak_hao_kheer', 'cham_cham', 'chana_masala',
 'chapati', 'chicken_pizza', 'chicken_razala', 'chicken_tikka',
 'chicken_tikka_masala', 'chicken_wings', 'chikki', 'chivda', 'chole_bhature',
 'daal_baati_churma', 'daal_puri', 'dabeli', 'dal_khichdi', 'dal_makhani',
 'dal_tadka', 'dharwad_pedha', 'dhokla', 'double_ka_meetha', 'dum_aloo',
 'falooda', 'fish_curry', 'gajar_ka_halwa', 'garlic_bread', 'gavvalu', 'ghevar',
 'grilled_sandwich', 'gujhia', 'gulab_jamun', 'hara_bhara_kabab', 'idiyappam',
 'idli', 'imarti', 'jalebi', 'kachori', 'kadai_paneer',
 'kadhi_pakoda', 'kaju_katli', 'kakinada_khaja', 'kalakand', 'karela_bharta',
 'khakhra', 'kheer', 'kofta',
 'kulfi', 'lassi', 'ledikeni', 'litti_chokha', 'lyangcha', 'maach_jhol',
 'makki_di_roti_sarson_da_saag', 'malpua', 'margherita_pizza', 'masala_dosa',
 'masala_papad', 'medu_vada', 'misal_pav', 'misi_roti', 'misti_doi',
 'modak', 'moong_dal_halwa', 'murukku', 'mysore_pak', 'naan',
 'navratan_korma', 'neer_dosa', 'onion_pakoda', 'palak_paneer', 'paneer_masala',
 'paneer_pizza', 'pani_puri', 'paniyaram', 'papdi_chaat', 'patrode',
 'pav_bhaji', 'pepperoni_pizza', 'phirni', 'pithe', 'poha',
 'pongal', 'poornalu', 'pootharekulu', 'puri_bhaji', 'qubani_ka_meetha',
 'rabri', 'rajma_chawal', 'ras_malai', 'rasgulla', 'rava_dosa',
 'sabudana_khichdi', 'sabudana_vada', 'samosa', 'sandesh', 'seekh_kebab',
 'set_dosa', 'sev_puri', 'shankarpali', 'sheer_korma', 'sheera',
 'shrikhand', 'soan_papdi', 'solkadhi', 'steamed_momo', 'sutar_feni',
 'thali', 'thukpa', 'unni_appam', 'uttapam', 'vada_pav']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

pre_process = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class FoodClassifierResNet(nn.Module):
    def __init__(self, num_classes=131, unfreeze_layers=0):
        super(FoodClassifierResNet, self).__init__()

        # load pre-trained ResNet-152
        self.base_model = models.resnet152(weights='IMAGENET1K_V2')

        # Freeze all base model parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze specified number of 'layer' blocks from the end
        # For ResNet, layer4 is the last convolutional block, then layer3, etc.
        # unfreeze_layers=1 -> unfreeze layer4
        # unfreeze_layers=2 -> unfreeze layer4, layer3
        # unfreeze_layers=3 -> unfreeze layer4, layer3, layer2
        # unfreeze_layers=4 -> unfreeze layer4, layer3, layer2, layer1
        named_layer_blocks = ['layer4', 'layer3', 'layer2', 'layer1']

        for i in range(min(unfreeze_layers, len(named_layer_blocks))):
            layer_name = named_layer_blocks[i]
            if hasattr(self.base_model, layer_name):
                layer = getattr(self.base_model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True

        # Extract features (everything up to the adaptive average pooling, excluding the original FC layer)
        # The `base_model`'s `avgpool` is included in `features_extractor`.
        self.features_extractor = nn.Sequential(
            self.base_model.conv1,
            self.base_model.bn1,
            self.base_model.relu,
            self.base_model.maxpool,
            self.base_model.layer1,
            self.base_model.layer2,
            self.base_model.layer3,
            self.base_model.layer4,
            self.base_model.avgpool # Include the original avgpool
        )

        # add custom output layer
        # The input features to the linear layer will be from the base_model's final feature map size after avgpool
        self.output_layer = nn.Linear(self.base_model.fc.in_features, num_classes) # ResNet-152 has 2048 features before FC

    def forward(self, x):
        x = self.features_extractor(x)
        x = torch.flatten(x, 1) # Flatten the (batch_size, 2048, 1, 1) to (batch_size, 2048)
        x = self.output_layer(x)
        return x

# use ../../models/food_resnet_v42_12_0.887.pth as path while testing predict.py
def resnet(img_file):
    rn_model = FoodClassifierResNet(unfreeze_layers=2)
    state_dict = torch.load('models/food_resnet_v42_12_0.887.pth', map_location='cpu')
    rn_model.load_state_dict(state_dict)
    rn_model.eval()

    # ../../static/uploads/{img_file}
    img = Image.open(img_file)
    x = pre_process(img)
    batch_t = torch.unsqueeze(x, 0).to(device)
    with torch.no_grad():
        output = rn_model(batch_t.to('cpu')).cpu().numpy()
        output = dict(zip( classes, list(map(float, output[0])) ))
        t5_preds = dict(sorted(output.items(), key=lambda x: x[1], reverse=True)[:5])
        t1_class = max(t5_preds, key=t5_preds.get)
        return {'t1_class': t1_class, 't5_preds': t5_preds}


def convns(img_file):
    session_cn = ort.InferenceSession('models/food_classifier_convnexts_v2.onnx')

    input_name = session_cn.get_inputs()[0].name
    output_name = session_cn.get_outputs()[0].name
    # print(input_name, output_name)

    img = Image.open(img_file)
    x = np.expand_dims(pre_process(img), axis=0)
    output = dict(zip( classes, list(map(float, session_cn.run([output_name], {input_name: x})[0][0])) ))
    t5_preds = dict(sorted(output.items(), key=lambda x: x[1], reverse=True)[:5])
    t1_class = max(t5_preds, key=t5_preds.get)
    
    return {'t1_class': t1_class, 't5_preds': t5_preds}
    

# print(resnet('misti_doi.jpg'))
# print(convns('misti_doi.jpg'))