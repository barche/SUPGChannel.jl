using SUPGChannel
using Test


# @time channelflow()
nls = SUPGChannel.DummyNLS(Ref{Any}())
channelflow(;nls)

