class Tensorkiko < Formula
  include Language::Python::Virtualenv

  desc "A fast and intuitive tool for visualizing and analyzing model structures from safetensors files"
  homepage "https://github.com/takara-ai/TensorKiko"
  url "https://github.com/takara-ai/TensorKiko/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "replace_with_actual_sha256_of_your_tarball"
  license "MIT"

  depends_on "python@3.11"

  def install
    virtualenv_create(libexec, "python3.11")
    virtualenv_install_with_resources
  end

  test do
    assert_match "TensorKiko", shell_output("#{bin}/tensorkiko --help")
  end
end