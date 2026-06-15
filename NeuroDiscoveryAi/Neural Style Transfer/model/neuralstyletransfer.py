import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys

# Setup GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Preprocessing
# ==========================================

def select_image(prompt_title="Select an Image"):
    """Opens a native file dialog to select an image."""
    root = tk.Tk()
    root.withdraw() # Hide the main empty tkinter window
    root.attributes('-topmost', True) # Force window to the front
    
    file_path = filedialog.askopenfilename(
        title=prompt_title,
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")] 
    )
    return file_path
    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We use the same normalization statistics here.
def load_image(image_path, max_size=512):
    """Loads, resizes, and normalizes an image."""
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    return image

def denormalize_image(tensor):
    """Reverts normalization for saving the final image."""
    image = tensor.cpu().clone().detach().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    return image.clamp(0, 1)


# Model Architecture
# ==========================================

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        #pre-trained features
        self.vgg_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        #(conv5_1), slice the model
        self.vgg_features = self.vgg_features[:29]
        
        # Freeze weights so they don't update
        for param in self.vgg_features.parameters():
            param.requires_grad = False

    def forward(self, x):
        style_features = []
        content_feature = None
        
        # Loop through the network 
        for layer_num, layer in enumerate(self.vgg_features):
            x = layer(x)
            
            # outputs at the specific layers
            if layer_num == 0:      # conv1_1
                style_features.append(x)
            elif layer_num == 5:    # conv2_1
                style_features.append(x)
            elif layer_num == 10:   # conv3_1
                style_features.append(x)
            elif layer_num == 19:   # conv4_1
                style_features.append(x)
            elif layer_num == 21:   # conv4_2
                content_feature = x 
            elif layer_num == 28:   # conv5_1
                style_features.append(x)
                
        return style_features, content_feature


# 3. Loss Functions
# ==========================================

def calc_gram_matrix(tensor):
    b, c, h, w = tensor.size()

    features = tensor.view(c, h * w)

    gram = torch.mm(features, features.t())

    gram = gram / (c * h * w)

    return gram

def content_loss(generated_features, target_features):
    """Mean Squared Error between deep features."""
    return torch.mean((generated_features - target_features) ** 2)

def style_loss(generated_features, style_gram_matrix):
    """Mean Squared Error between Gram matrices."""
    channels = generated_features.shape[1]
    height = generated_features.shape[2]
    width = generated_features.shape[3]
    
    generated_gram = calc_gram_matrix(generated_features)
    return torch.mean(
    (generated_gram - style_gram_matrix) ** 2
)


#  Optimization Loop 
# ==========================================

def run_style_transfer(content_img, style_img, num_steps=1500):
    model = VGG().to(device).eval()
    
    #targets
    target_style_features, _ = model(style_img)
    _, target_content_feature = model(content_img)
    
    # Gram matrices 
    target_style_grams = []
    for feature in target_style_features:
        gram = calc_gram_matrix(feature)
        target_style_grams.append(gram)
        
    #  Create the image
    generated_img = content_img.clone().requires_grad_(True).to(device)
    #generated_img = torch.randn_like(content_img).requires_grad_(True)
    
    # Adam Optimizer
    optimizer = optim.Adam([generated_img], lr=0.01)
    
    style_weight = 1e5
    content_weight = 1
    
    print("\nStarting optimization loop with Adam optimizer...")
    
    for step in range(num_steps):
        generated_style_features, generated_content_feature = model(generated_img)
        
        c_loss = content_loss(generated_content_feature, target_content_feature)
        
        s_loss = 0
        for i in range(len(generated_style_features)):
            gen_feat = generated_style_features[i]
            target_gram = target_style_grams[i]
            s_loss += style_loss(gen_feat, target_gram)
            
        total_loss = (content_weight * c_loss) + (style_weight * s_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            generated_img.clamp_(-2.1, 2.6)
            
        if step % 100 == 0:
            
            print(f"Step {step}:")
            print(f"  Raw Content Loss: {c_loss.item():.4f}")
            print(f"  Raw Style Loss:   {s_loss.item():.4f}")
            print(f"  Total Weighted:   {total_loss.item():.4f}")
            save_image(
                    denormalize_image(generated_img),
                    f"debug_{step}.jpg"
)
            
    return generated_img


#  Main 
# ==========================================

if __name__ == "__main__":
    print("Opening file explorer... Please select your Content (Original) Image.")
    content_path = select_image("Select Content Image")
    
    print("Opening file explorer... Please select your Style Image.")
    style_path = select_image("Select Style Image")

    if not content_path or not style_path:
        print("Execution cancelled: Both a content and style image are required.")
        sys.exit()

    print(f"\nLoading Content Image: {content_path.split('/')[-1]}")
    print(f"Loading Style Image: {style_path.split('/')[-1]}")

    content_img = load_image(content_path)
    style_img = load_image(style_path)

    final_tensor = run_style_transfer(content_img, style_img, num_steps=1500)
    
    print("\nOptimization complete. Saving output...")
    final_image = denormalize_image(final_tensor)
    output_filename = "output_styled_image1.jpg"
    
    save_image(final_image, output_filename)
    print(f"Success! Saved locally as '{output_filename}'.")