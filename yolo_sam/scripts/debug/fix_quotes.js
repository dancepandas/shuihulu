const fs = require('fs');
let content = fs.readFileSync('D:/chengs/9.project/shuihulu/doc/generate_doc.js', 'utf-8');

// Count occurrences
const leftCount = (content.match(/“/g) || []).length;
const rightCount = (content.match(/”/g) || []).length;
console.log(`Left curly quotes ("): ${leftCount}`);
console.log(`Right curly quotes ("): ${rightCount}`);

// Replace Chinese curly double quotes with regular double-quote text markers
content = content.replace(/“/g, "'");
content = content.replace(/”/g, "'");

// Verify
const afterLeft = (content.match(/“/g) || []).length;
const afterRight = (content.match(/”/g) || []).length;
console.log(`After fix - left: ${afterLeft}, right: ${afterRight}`);

fs.writeFileSync('D:/chengs/9.project/shuihulu/doc/generate_doc.js', content, 'utf-8');
console.log('Fixed and saved.');
