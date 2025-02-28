from diffusers import DiTTransformer2DModel


class DiT(DiTTransformer2DModel):
    def forward(self, inputs, time, context, **kwargs):
        return super().forward(inputs, time, context, **kwargs)[0]

