#!/bin/bash

# This scripts installs Lua, LuaRocks, and some Lua libraries on macOS.
# The main purpose is to install Busted for testing Neovim plugins.
# After the installation, you will be able to run test using busted:
#   busted --lua nlua spec/mytest_spec.lua

################################################################################
# Dependencies
################################################################################

xcode-select --install

# Lua Directory: where Lua and Luarocks will be installed
# You can change installation location by changing this variable
LUA_DIR="$HOME/Developer/lua"

mkdir -p $LUA_DIR

################################################################################
# Lua
################################################################################

# Download and Extract Lua Sources
cd /tmp
rm -rf lua-5.1.5.*
wget https://www.lua.org/ftp/lua-5.1.5.tar.gz
LUA_SHA='2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333'
shasum -a 256 lua-5.1.5.tar.gz | grep -q $LUA_SHA && echo "Hash matches" || echo "Hash don't match"
tar xvf lua-5.1.5.tar.gz
cd lua-5.1.5/

# Modify Makefile to set destination dir
sed -i '' "s#/usr/local#${LUA_DIR}/#g" Makefile

# Compile and install Lua
make macosx
make test && make install

# Export PATHs
export PATH="$PATH:$LUA_DIR/bin"
export LUA_CPATH="$LUA_DIR/lib/lua/5.1/?.so"
export LUA_PATH="$LUA_DIR/share/lua/5.1/?.lua;;"
export MANPATH="$LUA_DIR/share/man:$MANPATH"

# Verify Lua Installation
which lua
echo "Expected Output:"
echo "  ${LUA_DIR}/bin/lua"
lua -v
echo 'Expected Output:'
echo '  Lua 5.1.5  Copyright (C) 1994-2012 Lua.org, PUC-Rio'
file ${LUA_DIR}/bin/lua
echo "Expected Output (on Apple Silicon):"
echo "  ${LUA_DIR}/bin/lua: Mach-O 64-bit executable arm64"