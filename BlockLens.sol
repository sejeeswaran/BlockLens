// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BlockLensRegistry {
    struct Verdict {
        string status;
        string geminiVerdict;
        string blockLensVerdict;
        string supportingSignals;
        uint256 confidence;
        uint256 timestamp;
        address registrar;
    }

    mapping(bytes32 => Verdict) public verdicts;

    event VerdictRegistered(
        bytes32 indexed imageHash,
        string status,
        string geminiVerdict,
        string blockLensVerdict,
        uint256 confidence,
        address indexed registrar,
        uint256 timestamp
    );

    function registerVerdict(
        bytes32 _imageHash,
        string memory _status,
        string memory _geminiVerdict,
        string memory _blockLensVerdict,
        string memory _supportingSignals,
        uint256 _confidence
    ) public {
        require(
            verdicts[_imageHash].timestamp == 0,
            "Verdict already exists for this image."
        );

        verdicts[_imageHash] = Verdict({
            status: _status,
            geminiVerdict: _geminiVerdict,
            blockLensVerdict: _blockLensVerdict,
            supportingSignals: _supportingSignals,
            confidence: _confidence,
            timestamp: block.timestamp,
            registrar: msg.sender
        });

        emit VerdictRegistered(
            _imageHash,
            _status,
            _geminiVerdict,
            _blockLensVerdict,
            _confidence,
            msg.sender,
            block.timestamp
        );
    }

    function getVerdict(
        bytes32 _imageHash
    )
        public
        view
        returns (
            string memory status,
            string memory geminiVerdict,
            string memory blockLensVerdict,
            string memory supportingSignals,
            uint256 confidence,
            uint256 timestamp,
            address registrar
        )
    {
        Verdict memory v = verdicts[_imageHash];
        return (v.status, v.geminiVerdict, v.blockLensVerdict, v.supportingSignals, v.confidence, v.timestamp, v.registrar);
    }
}
