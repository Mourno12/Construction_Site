/**
 * auth.js
 * Password hashing (bcryptjs - pure JS, no native build step) and JWT
 * issuing/verification for the register/login/logout flow.
 */

const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'siteguard-dev-secret-change-me';
const JWT_EXPIRES_IN = '12h';

if (!process.env.JWT_SECRET) {
  console.warn(
    '[auth] JWT_SECRET is not set - using an insecure default. ' +
    'Set the JWT_SECRET environment variable before deploying this anywhere real.'
  );
}

async function hashPassword(plain) {
  return bcrypt.hash(plain, 10);
}

async function verifyPassword(plain, hash) {
  return bcrypt.compare(plain, hash);
}

function issueToken(user) {
  return jwt.sign(
    { sub: user.id, username: user.username, role: user.role },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES_IN }
  );
}

function verifyToken(token) {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (e) {
    return null;
  }
}

module.exports = { hashPassword, verifyPassword, issueToken, verifyToken };
