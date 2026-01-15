import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleCNN(nn.Module):
    def __init__(
        self, num_classes=10, input_height=28, input_width=28, input_channels=1, output_channels=32
    ):
        super(SimpleCNN, self).__init__()

        # Convolutional Block 1
        # Input: 1 channel (grayscale), Output: 16 channels, Kernel: 3x3
        # Padding=1 maintains spatial dimensions (28x28 -> 28x28) before pooling

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.H, self.W = self.calc_conv_output_shape(
            input_height, input_width, kernel_size=3, padding=1
        )

        self.H, self.W = self.calc_conv_output_shape(self.H, self.W, kernel_size=2, stride=2)
        # Convolutional Block 2
        # Input: 16 channels, Output: 32 channels, Kernel: 3x3
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=output_channels, kernel_size=3, padding=1
        )
        self.H, self.W = self.calc_conv_output_shape(self.H, self.W, kernel_size=3, padding=1)
        # Max Pooling layer (2x2) to downsample features by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.H, self.W = self.calc_conv_output_shape(self.H, self.W, kernel_size=2, stride=2)
        # Fully Connected (MLP) Layer
        # The input features will be flattened.
        # Calculation: 32 channels * 7 height * 7 width = 1568 features

        self.fc1 = nn.Linear(output_channels * self.H * self.W, num_classes)

    def forward(self, x):
        # --- Block 1 ---
        # Input shape:  [Batch, 1, 28, 28]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        # After Conv1 (padded) -> [Batch, 16, 28, 28]
        # After Pool1 (2x2)    -> [Batch, 16, 14, 14]

        # --- Block 2 ---
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        # After Conv2 (padded) -> [Batch, 32, 14, 14]
        # After Pool2 (2x2)    -> [Batch, 32, 7, 7]

        # --- Flattening ---
        # Reshape for the linear layer: collapse (Channel, Height, Width)
        x = x.view(x.size(0), -1)
        # Shape: [Batch, 1568] (where 1568 = 32 * 7 * 7)

        # --- Classifier ---
        x = self.fc1(x)
        # Output Shape: [Batch, num_classes] (Logits)

        return x

    @staticmethod
    def calc_conv_output_shape(h_in, w_in, kernel_size, stride=1, padding=0, dilation=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        # floor( (H_in + 2*padding - dilation*(kernel - 1) - 1) / stride + 1 )
        # floor( (H_in + 2*padding - kernel) / stride + 1 )
        h_out = math.floor(
            (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
        )

        # (Width)
        w_out = math.floor(
            (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
        )

        return h_out, w_out


def print_layer_transition(h_in, w_in, c_in, c_out, kernel_size, stride, padding):
    h_out, w_out = SimpleCNN.calc_conv_output_shape(h_in, w_in, kernel_size, stride, padding)

    print(f"输入 Tensor 形状: (Batch, {c_in}, {h_in}, {w_in})")
    print(
        f"参数设置: Kernel={kernel_size}, Stride={stride}, Padding={padding}, Out_Channels={c_out}"
    )
    print(f"输出 Tensor 形状: (Batch, {c_out}, {h_out}, {w_out})")
    print("-" * 30)


# --- Example Usage ---
if __name__ == "__main__":
    # Create the model instance
    model = SimpleCNN(num_classes=10)

    # Create a dummy input tensor (Batch Size=64, Channels=1, Height=28, Width=28)
    dummy_input = torch.randn(64, 1, 28, 28)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [64, 10]

    # Print layer transitions
    print_layer_transition(h_in=28, w_in=28, c_in=1, c_out=16, kernel_size=3, stride=1, padding=1)

    # 示例 2: 下采样 (尺寸减半)
    # 28x28, k=3, s=2, p=1
    print_layer_transition(h_in=28, w_in=28, c_in=16, c_out=32, kernel_size=3, stride=2, padding=1)

    # 示例 3: 不使用 Padding (尺寸减小)
    # 28x28, k=5, s=1, p=0
    print_layer_transition(h_in=28, w_in=28, c_in=1, c_out=6, kernel_size=5, stride=1, padding=0)
