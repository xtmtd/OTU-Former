from otuformer.training.scheduler import (
    cosine_lr_schedule,
    ema_momentum_schedule,
    teacher_temp_schedule,
)


def test_cosine_lr_at_boundaries():
    lr = cosine_lr_schedule(
        epoch=0, max_epochs=100, base_lr=1e-3, min_lr=1e-6, warmup_epochs=5
    )
    assert lr < 1e-3
    lr_end = cosine_lr_schedule(
        epoch=99, max_epochs=100, base_lr=1e-3, min_lr=1e-6, warmup_epochs=5
    )
    assert abs(lr_end - 1e-6) < 1e-7


def test_ema_momentum_increases():
    m0 = ema_momentum_schedule(step=0, total_steps=1000, start=0.995, end=0.999)
    m1 = ema_momentum_schedule(step=999, total_steps=1000, start=0.995, end=0.999)
    assert m1 > m0
    assert m1 <= 0.999


def test_teacher_temp_warmup():
    t0 = teacher_temp_schedule(epoch=0, warmup_epochs=10, start=0.04, end=0.07)
    t5 = teacher_temp_schedule(epoch=5, warmup_epochs=10, start=0.04, end=0.07)
    t10 = teacher_temp_schedule(epoch=10, warmup_epochs=10, start=0.04, end=0.07)
    assert t0 < t5 < t10
    assert abs(t10 - 0.07) < 1e-6
