const os = require('node:os');

const originalNetworkInterfaces = os.networkInterfaces;

os.networkInterfaces = function networkInterfacesWithLoopbackFallback() {
  try {
    return originalNetworkInterfaces.call(os);
  } catch (error) {
    if (!String(error && error.message).includes('uv_interface_addresses')) {
      throw error;
    }
    return {
      lo: [
        {
          address: '127.0.0.1',
          netmask: '255.0.0.0',
          family: 'IPv4',
          mac: '00:00:00:00:00:00',
          internal: true,
          cidr: '127.0.0.1/8',
        },
        {
          address: '::1',
          netmask: 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff',
          family: 'IPv6',
          mac: '00:00:00:00:00:00',
          internal: true,
          cidr: '::1/128',
          scopeid: 0,
        },
      ],
    };
  }
};
