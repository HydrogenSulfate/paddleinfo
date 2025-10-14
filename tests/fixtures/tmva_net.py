import paddle


class DoubleConvBlock(paddle.nn.Layer):
    """(2D conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = paddle.nn.Sequential(
            paddle.nn.Conv2d(
                in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil
            ),
            paddle.nn.BatchNorm2D(num_features=out_ch),
            paddle.nn.LeakyReLU(),
            paddle.nn.Conv2d(
                out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil
            ),
            paddle.nn.BatchNorm2D(num_features=out_ch),
            paddle.nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(paddle.nn.Layer):
    """(3D conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = paddle.nn.Sequential(
            paddle.nn.Conv3d(
                in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil
            ),
            paddle.nn.BatchNorm3D(num_features=out_ch),
            paddle.nn.LeakyReLU(),
            paddle.nn.Conv3d(
                out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil
            ),
            paddle.nn.BatchNorm3D(num_features=out_ch),
            paddle.nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(paddle.nn.Layer):
    """(2D conv => BN => LeakyReLU)"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = paddle.nn.Sequential(
            paddle.nn.Conv2d(
                in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil
            ),
            paddle.nn.BatchNorm2D(num_features=out_ch),
            paddle.nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ASPPBlock(paddle.nn.Layer):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = paddle.nn.AvgPool2D(
            kernel_size=(64, 64), exclusive=False
        )
        self.conv1_1x1 = paddle.nn.Conv2d(
            in_ch, out_ch, kernel_size=1, padding=0, dilation=1
        )
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = paddle.nn.functional.interpolate(
            x=self.global_avg_pool(x),
            size=(64, 64),
            align_corners=False,
            mode="bilinear",
        )
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = paddle.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(paddle.nn.Layer):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type):
        super().__init__()
        self.signal_type = signal_type
        self.double_3dconv_block1 = Double3DConvBlock(
            in_ch=1, out_ch=128, k_size=3, pad=(0, 1, 1), dil=1
        )
        self.doppler_max_pool = paddle.nn.MaxPool2D(kernel_size=2, stride=(2, 1))
        self.max_pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.double_conv_block2 = DoubleConvBlock(
            in_ch=128, out_ch=128, k_size=3, pad=1, dil=1
        )
        self.single_conv_block1_1x1 = ConvBlock(
            in_ch=128, out_ch=128, k_size=1, pad=0, dil=1
        )

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = paddle.squeeze(x=x1, axis=2)
        if self.signal_type in ("range_doppler", "angle_doppler"):
            x1_pad = paddle.compat.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1_down = self.doppler_max_pool(x1_pad)
        else:
            x1_down = self.max_pool(x1)
        x2 = self.double_conv_block2(x1_down)
        if self.signal_type in ("range_doppler", "angle_doppler"):
            x2_pad = paddle.compat.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)
        x3 = self.single_conv_block1_1x1(x2_down)
        return x2_down, x3


class TMVANet_Encoder(paddle.nn.Layer):
    """
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.ra_encoding_branch = EncodingBranch("range_angle")
        self.rd_encoding_branch = EncodingBranch("range_doppler")
        self.ad_encoding_branch = EncodingBranch("angle_doppler")
        self.rd_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ad_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.rd_single_conv_block1_1x1 = ConvBlock(
            in_ch=640, out_ch=128, k_size=1, pad=0, dil=1
        )
        self.ra_single_conv_block1_1x1 = ConvBlock(
            in_ch=640, out_ch=128, k_size=1, pad=0, dil=1
        )
        self.ad_single_conv_block1_1x1 = ConvBlock(
            in_ch=640, out_ch=128, k_size=1, pad=0, dil=1
        )

    def forward(self, x_rd, x_ra, x_ad, printshape=False):
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent = self.ad_encoding_branch(x_ad)
        x1_rd = self.rd_aspp_block(rd_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x1_ad = self.ad_aspp_block(ad_features)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad = self.ad_single_conv_block1_1x1(x1_ad)
        x3 = paddle.cat((rd_latent, ra_latent, ad_latent), 1)
        return x3, x2_rd, x2_ad, x2_ra


class TMVANet_Decoder(paddle.nn.Layer):
    """
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.rd_single_conv_block2_1x1 = ConvBlock(
            in_ch=384, out_ch=128, k_size=1, pad=0, dil=1
        )
        self.ra_single_conv_block2_1x1 = ConvBlock(
            in_ch=384, out_ch=128, k_size=1, pad=0, dil=1
        )
        self.rd_upconv1 = paddle.nn.Conv2DTranspose(
            in_channels=384, out_channels=128, kernel_size=(2, 1), stride=(2, 1)
        )
        self.ra_upconv1 = paddle.nn.Conv2DTranspose(
            in_channels=384, out_channels=128, kernel_size=2, stride=2
        )
        self.rd_double_conv_block1 = DoubleConvBlock(
            in_ch=128, out_ch=128, k_size=3, pad=1, dil=1
        )
        self.ra_double_conv_block1 = DoubleConvBlock(
            in_ch=128, out_ch=128, k_size=3, pad=1, dil=1
        )
        self.rd_upconv2 = paddle.nn.Conv2DTranspose(
            in_channels=128, out_channels=128, kernel_size=(2, 1), stride=(2, 1)
        )
        self.ra_upconv2 = paddle.nn.Conv2DTranspose(
            in_channels=128, out_channels=128, kernel_size=2, stride=2
        )
        self.rd_double_conv_block2 = DoubleConvBlock(
            in_ch=128, out_ch=128, k_size=3, pad=1, dil=1
        )
        self.ra_double_conv_block2 = DoubleConvBlock(
            in_ch=128, out_ch=128, k_size=3, pad=1, dil=1
        )
        self.rd_final = paddle.nn.Conv2d(
            in_channels=128, out_channels=n_classes, kernel_size=1
        )
        self.ra_final = paddle.nn.Conv2d(
            in_channels=128, out_channels=n_classes, kernel_size=1
        )

    def forward(self, x3, x2_rd, x2_ad, x2_ra):
        x3_rd = self.rd_single_conv_block2_1x1(x3)
        x3_ra = self.ra_single_conv_block2_1x1(x3)
        x4_rd = paddle.cat((x2_rd, x3_rd, x2_ad), 1)
        x4_ra = paddle.cat((x2_ra, x3_ra, x2_ad), 1)
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)
        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)
        return x9_rd, x9_ra


class TMVANet(paddle.nn.Layer):
    """
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.encoder = TMVANet_Encoder(n_classes, n_frames)
        self.decoder = TMVANet_Decoder(n_classes, n_frames)

    def forward(self, x_rd, x_ra, x_ad):
        x3, x2_rd, x2_ad, x2_ra = self.encoder(x_rd, x_ra, x_ad)
        x9_rd, x9_ra = self.decoder(x3, x2_rd, x2_ad, x2_ra)
        return x9_rd, x9_ra
