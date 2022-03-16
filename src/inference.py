import torch
import utils
from model import build_model

def predict(img, model, lbl_encoder, device = "cpu", max_len = 5, label=""):
    
    model.eval()
    with torch.no_grad():
        encoded,_ = model(img, None, trg_mask = None, is_inference=True)        
        
    L = [lbl_encoder.transform(["<"])[0]]   
    
    predicts = [lbl_encoder.transform(["<"])[0]]

    result = []

    for i in range(5):
        with torch.no_grad():         
            # for k in range(6 - len(L)):
            #     L.append(0)   
            trg_tensor = torch.LongTensor(predicts).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            # trg_mask=None
            output, _ = model.decoder( trg_tensor, encoded,trg_mask = trg_mask, src_mask=None)
            predicted = output.argmax(2)
            # idx = L.index(0) - 1
            idx=-1
            predicted = predicted[:,idx].item()

            predicts.append(predicted)            
            
            abc = lbl_encoder.classes_[predicted]
            result.append(abc)
            eos = lbl_encoder.transform([">"])[0]
            if predicted == eos:
                break
    return "".join(result)


train_loader, test_loader, lbl_encoder, test_orig_targets = utils.prepare_data()
model = build_model("/home/hung/learn/pytorch/captcha_trainsformer/weights/model_326.pt")
print(model)
d = 0
for bindex, data in enumerate(test_loader):
    
    for j in range(8):
        img = data["images"][j].unsqueeze(0)
        lbl = data["raw_targets"][j]
        lbl = lbl.replace("<", "")
        lbl = lbl.replace(">", "")
        res = predict(img, model, lbl_encoder, "cpu", 5, lbl)
        print(f"{res}, {lbl}")
        if res != lbl:
            d+=1
print(f"False predict {d}")    
